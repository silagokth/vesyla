#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <optional>

#include "vesyla/Dialect/Pasm/Transforms/GenerateIcdepPass.hpp"
#include "vesyla/Support/Config.hpp"

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

// Read the producer's `delay` attribute. Returns nullopt when it is absent or
// not an integer; the caller treats that as a hard error. The delay is carried
// on the originating drra.rop and copied onto the icdep anchors built from it.
std::optional<int32_t> get_producer_delay(mlir::Operation *op) {
  if (auto delay = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
          op->getAttr("delay"))) {
    return static_cast<int32_t>(delay.getInt());
  }
  return std::nullopt;
}

// Build the first/last anchor for a producer. `first` uses each enclosing
// loop's lower bound; `last` uses each upper bound minus one. Top-level
// producers (no enclosing loops) get a bare anchor with no event/indices. The
// `delay` is taken from the originating drra.rop.
AnchorAttr build_anchor(mlir::MLIRContext *ctx, mlir::FlatSymbolRefAttr instr,
                        llvm::ArrayRef<mlir::affine::AffineForOp> loops,
                        bool is_last, int32_t delay) {
  llvm::SmallVector<int32_t> idx;
  for (mlir::affine::AffineForOp loop : loops) {
    if (is_last) {
      int64_t ub =
          loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 0;
      idx.push_back(static_cast<int32_t>(ub - 1));
    } else {
      int64_t lb =
          loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
      idx.push_back(static_cast<int32_t>(lb));
    }
  }
  llvm::StringRef event = loops.empty() ? "" : "e0";
  return AnchorAttr::get(ctx, instr, event, idx, delay);
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
        std::optional<int32_t> delay = get_producer_delay(producer);
        if (!delay) {
          producer->emitError(
              "icdep producer is missing an integer 'delay' attribute: ")
              << *producer;
          signalPassFailure();
          return mlir::WalkResult::interrupt();
        }
        AnchorAttr first =
            build_anchor(ctx, instr, loops, /*is_last=*/false, *delay);
        AnchorAttr last =
            build_anchor(ctx, instr, loops, /*is_last=*/true, *delay);

        for (mlir::OpResult result : producer->getResults()) {
          ResourceAttr src =
              get_result_resource(producer, result.getResultNumber());
          if (!src) {
            continue;
          }
          // Merge every resourced consumer of this value into a single
          // icdep's dst array (consumers may be nested in loops).
          llvm::SmallVector<mlir::Attribute> dst;
          for (mlir::OpOperand &use : result.getUses()) {
            mlir::Operation *consumer = use.getOwner();
            if (!consumer->hasAttr("resource")) {
              continue;
            }
            ResourceAttr dres =
                get_operand_resource(consumer, use.getOperandNumber());
            if (dres) {
              dst.push_back(dres);
            }
          }
          if (dst.empty()) {
            continue;
          }
          // Derive the data kind (word/bulk) from the source port via the
          // fabric config's port table; dir is left empty here.
          Config cfg;
          std::string kind = cfg.get_port_info(src.getPort()).kind;
          IcDepOp::create(builder, producer->getLoc(), src,
                          builder.getArrayAttr(dst),
                          builder.getStringAttr(kind), first, last,
                          /*dir=*/mlir::StringAttr());
        }
      }
      return mlir::WalkResult::advance();
    });
    (void)result;
  }
};

} // namespace
} // namespace vesyla::pasm
