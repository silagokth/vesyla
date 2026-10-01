#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

#include "vesyla/Dialect/Pasm/Transforms/GenerateIcdepPass.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_GENERATEICDEPPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Read the resource for a given result of a producer op. Ops with a single
// endpoint carry a plain ResourceAttr; ops with multiple inputs and an output
// carry an array laid out as [outputs..., inputs...], so result `r` lives at
// index `r`.
ResourceAttr get_result_resource(mlir::Operation *op, unsigned result_index) {
  mlir::Attribute attr = op->getAttr("resource");
  if (auto res = mlir::dyn_cast_or_null<ResourceAttr>(attr)) {
    return res;
  }
  if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
    if (result_index < arr.size()) {
      return mlir::dyn_cast<ResourceAttr>(arr[result_index]);
    }
  }
  return nullptr;
}

// Read the resource for a given operand of a consumer op. In the array layout
// [outputs..., inputs...], operand `o` lives at index `num_results + o`.
ResourceAttr get_operand_resource(mlir::Operation *op,
                                  unsigned operand_index) {
  mlir::Attribute attr = op->getAttr("resource");
  if (auto res = mlir::dyn_cast_or_null<ResourceAttr>(attr)) {
    return res;
  }
  if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
    unsigned idx = op->getNumResults() + operand_index;
    if (idx < arr.size()) {
      return mlir::dyn_cast<ResourceAttr>(arr[idx]);
    }
  }
  return nullptr;
}

// Forward-traverse the SSA value `value`, collecting the resource of every
// resourced consumer reachable from it. Resourced consumers terminate a chain.
// affine.for iter-args and affine.yield carry no resource attribute, so they are
// "looked through" to follow loop-carried data flow:
//   - a value used as an affine.for iter-arg init flows into the matching region
//     iter-arg (iteration 0 and, via the yield back-edge, every later one);
//   - a value yielded by affine.yield flows into both the matching loop result
//     (the final value) and the matching region iter-arg (the next iteration).
// `visited` guards against the iter-arg <-> yield cycle.
void collect_consumer_resources(mlir::Value value,
                                llvm::SmallVectorImpl<mlir::Attribute> &dst,
                                llvm::SmallPtrSetImpl<void *> &visited) {
  auto follow = [&](mlir::Value next) {
    if (visited.insert(next.getAsOpaquePointer()).second) {
      collect_consumer_resources(next, dst, visited);
    }
  };

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *consumer = use.getOwner();
    if (consumer->hasAttr("resource")) {
      ResourceAttr dres =
          get_operand_resource(consumer, use.getOperandNumber());
      if (dres) {
        dst.push_back(dres);
      }
      continue;
    }
    if (auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(consumer)) {
      unsigned ctrl = loop.getNumControlOperands();
      if (use.getOperandNumber() >= ctrl) {
        unsigned i = use.getOperandNumber() - ctrl;
        if (i < loop.getRegionIterArgs().size()) {
          follow(loop.getRegionIterArgs()[i]);
        }
      }
      continue;
    }
    if (auto yield = mlir::dyn_cast<mlir::affine::AffineYieldOp>(consumer)) {
      if (auto loop =
              mlir::dyn_cast<mlir::affine::AffineForOp>(yield->getParentOp())) {
        unsigned i = use.getOperandNumber();
        if (i < loop.getNumResults()) {
          follow(loop.getResult(i));
          follow(loop.getRegionIterArgs()[i]);
        }
      }
      continue;
    }
  }
}

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

// Read the producer's `delay` attribute, which is carried on the originating
// drra.rop and copied onto the icdep anchors built from it.
//
// An absent one reads as no delay. The component library does not describe how
// long an operation takes yet -- a resource's description says what it computes
// and which of its parts it holds, not its latency -- so nothing on the
// selection path writes a delay, and only hand-written input carries one.
// CreateConstraintsPass already defaults the same way.
int32_t get_producer_delay(mlir::Operation *op) {
  if (auto delay = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
          op->getAttr("delay"))) {
    return static_cast<int32_t>(delay.getInt());
  }
  return 0;
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
  size_t depth = 0;
  for (mlir::affine::AffineForOp loop : loops) {
    if (is_last && depth >= pass) {
      int64_t ub =
          loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 0;
      idx.push_back(static_cast<int32_t>(ub - 1));
    } else {
      int64_t lb =
          loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
      idx.push_back(static_cast<int32_t>(lb));
    }
    ++depth;
  }
  // Loop dimensions map to IR indices; MT (event id) is 0 and there is no OR.
  return AnchorAttr::get(ctx, instr, /*or_idx=*/{}, /*mt=*/0, idx, delay);
}

// The data kind a value travels as: a vector is a bulk transfer, a scalar a
// word transfer.
llvm::StringRef kind_of(mlir::Value value) {
  return mlir::isa<mlir::ShapedType>(value.getType()) ? "bulk" : "word";
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

      for (mlir::Operation *producer : producers) {
        auto instr =
            mlir::dyn_cast<mlir::FlatSymbolRefAttr>(producer->getAttr("id"));
        if (!instr) {
          continue;
        }
        llvm::SmallVector<mlir::affine::AffineForOp> loops =
            get_enclosing_loops(producer, epoch);
        int32_t delay = get_producer_delay(producer);
        AnchorAttr first =
            build_anchor(ctx, instr, loops, /*is_last=*/false, delay);

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
              build_anchor(ctx, instr, loops, /*is_last=*/true, delay, pass);
          llvm::SmallVector<int32_t> pass_hi;
          for (size_t depth = 0; depth < pass; ++depth) {
            int64_t ub = loops[depth].hasConstantUpperBound()
                             ? loops[depth].getConstantUpperBound()
                             : 1;
            pass_hi.push_back(static_cast<int32_t>(ub - 1));
          }
          auto mark = [&](IcDepOp icdep) {
            if (pass > 0) {
              icdep->setAttr("pass_hi", builder.getDenseI32ArrayAttr(pass_hi));
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
    (void)result;
  }
};

} // namespace
} // namespace vesyla::pasm
