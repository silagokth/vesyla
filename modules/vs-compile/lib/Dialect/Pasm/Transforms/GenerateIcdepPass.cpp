#include "vesyla/Support/LoopShape.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/Transforms/GenerateIcdepPass.hpp"
#include "vesyla/Dialect/Pasm/Transforms/ResourceFlow.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_GENERATEICDEPPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Collect the affine.for loops enclosing `op` up to (but excluding) `epoch`,
// ordered outermost to innermost.
llvm::SmallVector<mlir::affine::AffineForOp>
get_enclosing_loops(mlir::Operation *op, EpochOp epoch) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops;
  for (mlir::Operation *parent = op->getParentOp();
       parent && parent != epoch.getOperation();
       parent = parent->getParentOp()) {
    if (auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      loops.push_back(loop);
    }
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

// Read the producer's latency, which is carried on the originating drra.rop and
// copied onto the icdep anchors built from it: the cycles from a pass's event
// to its result being there to be taken. `delay` is the older name for it.
//
// An absent one reads as no delay, as CreateConstraintsPass also defaults.
int32_t get_producer_delay(mlir::Operation *op) {
  for (llvm::StringRef name : {"latency", "delay"}) {
    if (auto attr =
            mlir::dyn_cast_or_null<mlir::IntegerAttr>(op->getAttr(name))) {
      return static_cast<int32_t>(attr.getInt());
    }
  }
  return 0;
}

// The op a producer's transfer is timed by, and its delay. A conf-only rop has
// no event: use its first operand from an op in the same loops, adding the
// latencies; the producer itself when there is none.
std::pair<mlir::Operation *, int32_t> timing_source(mlir::Operation *producer,
                                                    EpochOp epoch) {
  mlir::Operation *op = producer;
  int32_t delay = get_producer_delay(producer);
  for (int hops = 0; hops < 16; ++hops) {
    auto rop = mlir::dyn_cast<drra::RopOp>(op);
    if (!rop || rop.getEvtAttr() || !rop.getConfAttr()) {
      return {op, delay};
    }
    llvm::SmallVector<mlir::affine::AffineForOp> loops =
        get_enclosing_loops(op, epoch);
    mlir::Operation *next = nullptr;
    for (mlir::Value operand : op->getOperands()) {
      mlir::Operation *def = operand.getDefiningOp();
      // Not into a loop: its results are the values it carries, not an event.
      if (def && def->hasAttr("id") &&
          !mlir::isa<mlir::affine::AffineForOp>(def) &&
          get_enclosing_loops(def, epoch) == loops) {
        next = def;
        break;
      }
    }
    if (!next) {
      break;
    }
    op = next;
    delay += get_producer_delay(op);
  }
  return {producer, get_producer_delay(producer)};
}

// Build the first/last anchor for a producer. `first` uses each enclosing
// loop's lower bound; `last` uses each upper bound minus one. Top-level
// producers (no enclosing loops) get a bare anchor with no event/indices. The
// `delay` is taken from the originating drra.rop.
//
// The outermost `pass` loops are pinned to their first iteration for `last`
// too: the transfer gives its route up at the end of each pass through them,
// so its last use is the end of the first pass, not of the whole nest.
AnchorAttr build_anchor(mlir::MLIRContext *ctx, mlir::FlatSymbolRefAttr instr,
                        llvm::ArrayRef<mlir::affine::AffineForOp> loops,
                        bool is_last, int32_t delay, size_t pass = 0) {
  llvm::SmallVector<int32_t> idx;
  llvm::SmallVector<int64_t> at;
  size_t depth = 0;
  for (mlir::affine::AffineForOp loop : loops) {
    if (is_last && depth >= pass) {
      // The last iteration in the pass the outer indices name (AGU Slope
      // Levels: a trip count may follow an outer loop).
      at.push_back(vesyla::util::trips_at(loop, at) - 1);
    } else {
      int64_t lb =
          loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
      at.push_back(lb);
    }
    idx.push_back(static_cast<int32_t>(at.back()));
    ++depth;
  }
  // Loop dimensions map to IR indices; MT (event id) is 0 and there is no OR.
  return AnchorAttr::get(ctx, instr, /*or_idx=*/{}, /*mt=*/0, idx, delay);
}

// How many of `loops`, outermost first, the transfer shares with another of
// its kind that needs a different route. Such a loop is one whose body
// switches between routes, so the transfer holds its route for one pass
// through it and takes it up again on the next; the loops inside hold this
// transfer alone and the route stays put across them. Zero when the transfer
// has the whole nest to itself.
//
// Two bulk transfers leaving the same slot share a route -- one send, with a
// receive that lists both destinations -- so they do not count against each
// other. Word transfers are all counted.
size_t pass_depth(mlir::Operation *producer, llvm::StringRef kind,
                  ResourceAttr src,
                  llvm::ArrayRef<mlir::affine::AffineForOp> loops,
                  llvm::ArrayRef<mlir::Operation *> producers) {
  for (size_t depth = loops.size(); depth > 0; --depth) {
    for (mlir::Operation *other : producers) {
      if (other == producer || !loops[depth - 1]->isProperAncestor(other)) {
        continue;
      }
      for (mlir::OpResult result : other->getResults()) {
        if (kind_of(result) != kind) {
          continue;
        }
        ResourceAttr other_src =
            get_result_resource(other, result.getResultNumber());
        if (kind == "bulk" && other_src && other_src.getRow() == src.getRow() &&
            other_src.getCol() == src.getCol() &&
            other_src.getSlot() == src.getSlot()) {
          continue;
        }
        return depth;
      }
    }
  }
  return 0;
}

class GenerateIcdepPass
    : public impl::GenerateIcdepPassBase<GenerateIcdepPass> {
public:
  using impl::GenerateIcdepPassBase<GenerateIcdepPass>::GenerateIcdepPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::WalkResult result = module.walk([&](EpochOp epoch) {
      mlir::Block &block = epoch.getBody().front();

      // Gather every resourced producer in the epoch, including ops nested
      // inside affine.for loops.
      llvm::SmallVector<mlir::Operation *> producers;
      epoch.walk([&](mlir::Operation *op) {
        if (op->getNumResults() > 0 && op->hasAttr("resource") &&
            op->hasAttr("id")) {
          producers.push_back(op);
        }
      });

      mlir::OpBuilder builder(ctx);
      builder.setInsertionPoint(block.getTerminator());

      int32_t pos = -1;
      for (mlir::Operation *producer : producers) {
        ++pos;
        auto instr =
            mlir::dyn_cast<mlir::FlatSymbolRefAttr>(producer->getAttr("id"));
        if (!instr) {
          continue;
        }
        llvm::SmallVector<mlir::affine::AffineForOp> loops =
            get_enclosing_loops(producer, epoch);
        auto [source, delay] = timing_source(producer, epoch);
        auto source_id =
            mlir::cast<mlir::FlatSymbolRefAttr>(source->getAttr("id"));
        AnchorAttr first =
            build_anchor(ctx, source_id, loops, /*is_last=*/false, delay);

        for (mlir::OpResult result : producer->getResults()) {
          ResourceAttr src =
              get_result_resource(producer, result.getResultNumber());
          if (!src) {
            continue;
          }
          // Collect every resourced consumer of this value. Consumers may be
          // nested in loops and reached through affine.for iter-args /
          // affine.yield back-edges.
          llvm::SmallVector<mlir::Attribute> dst;
          llvm::SmallPtrSet<void *, 8> visited;
          collect_consumer_resources(result, dst, visited);
          // Drop receivers on the same physical resource (row, col, slot) as
          // the source: such a dependency is internal to the module and needs
          // no interconnect routing.
          llvm::erase_if(dst, [&](mlir::Attribute attr) {
            auto dres = mlir::dyn_cast<ResourceAttr>(attr);
            return dres && dres.getRow() == src.getRow() &&
                   dres.getCol() == src.getCol() &&
                   dres.getSlot() == src.getSlot();
          });
          if (dst.empty()) {
            continue;
          }
          // Derive the data kind (word/bulk) from the routed value's type:
          // a scalar (i16) is a word transfer, a vector (vector<16xi16>) is a
          // bulk transfer. dir is left empty here.
          std::string kind = kind_of(result).str();
          // The loops the route is released across, and how far each runs, so
          // the binding can repeat over them what it finds for one pass.
          size_t pass = pass_depth(producer, kind, src, loops, producers);
          AnchorAttr last =
              build_anchor(ctx, source_id, loops, /*is_last=*/true, delay, pass);
          llvm::SmallVector<int32_t> pass_hi;
          llvm::SmallVector<mlir::Attribute> pass_loops;
          for (size_t depth = 0; depth < pass; ++depth) {
            pass_hi.push_back(static_cast<int32_t>(
                vesyla::util::trips_at_last(loops[depth]) - 1));
            auto id = loops[depth]->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
            if (!id) {
              loops[depth]->emitError("generate-icdep: loop has no id");
              return mlir::WalkResult::interrupt();
            }
            pass_loops.push_back(id);
          }
          auto mark = [&](IcDepOp icdep) {
            if (pass > 0) {
              icdep->setAttr("pass_hi", builder.getDenseI32ArrayAttr(pass_hi));
              icdep->setAttr("loops", builder.getArrayAttr(pass_loops));
              icdep->setAttr("pos", builder.getI32IntegerAttr(pos));
            }
          };
          if (kind == "bulk") {
            // bulk fans out: all receivers share one icdep.
            mark(IcDepOp::create(builder, producer->getLoc(), src,
                                 builder.getArrayAttr(dst),
                                 builder.getStringAttr(kind), first, last,
                                 /*dir=*/mlir::StringAttr()));
          } else {
            // word is point-to-point: one icdep per receiver.
            for (mlir::Attribute dres : dst) {
              mark(IcDepOp::create(builder, producer->getLoc(), src,
                                   builder.getArrayAttr(dres),
                                   builder.getStringAttr(kind), first, last,
                                   /*dir=*/mlir::StringAttr()));
            }
          }
        }
      }
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
