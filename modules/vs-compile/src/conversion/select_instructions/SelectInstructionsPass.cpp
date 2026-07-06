#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "SelectInstructionsPass.hpp"

// NativeHelpers.hpp must precede the generated .inc: the inline PDLL code blocks
// expand to free functions that call ::vesyla::sel::* by name.
#include "NativeHelpers.hpp"
#include "conversion/select_instructions/patterns.pdll.h.inc"

namespace vesyla::conversion::select_instructions {
#define GEN_PASS_DEF_SELECTINSTRUCTIONSPASS
#include "conversion/select_instructions/Passes.hpp.inc"

namespace {

// True for the upstream ops that this pass is responsible for selecting away.
// A survivor after the greedy rewrite means no pattern claimed it.
bool is_unselected_source(mlir::Operation *op) {
  return llvm::isa<mlir::arith::MulIOp, mlir::arith::AddIOp,
                   mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp,
                   mlir::affine::AffineVectorLoadOp,
                   mlir::affine::AffineVectorStoreOp>(op);
}

class SelectInstructionsPass
    : public impl::SelectInstructionsPassBase<SelectInstructionsPass> {
public:
  using impl::SelectInstructionsPassBase<
      SelectInstructionsPass>::SelectInstructionsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    if (mlir::failed(applyPatternsGreedily(
            module, mlir::FrozenRewritePatternSet(std::move(patterns))))) {
      signalPassFailure();
      return;
    }

    // affine.apply ops that only fed converted accesses are now dead; their
    // index maps were lifted onto the rops. Erase them so AffineToRepPass sees
    // loop bodies of only rops and nested loops.
    llvm::SmallVector<mlir::affine::AffineApplyOp> dead;
    module.walk([&](mlir::affine::AffineApplyOp apply) {
      if (apply.use_empty()) {
        dead.push_back(apply);
      }
    });
    for (mlir::affine::AffineApplyOp apply : dead) {
      apply.erase();
    }

    // Completeness gate: every source op must have been selected.
    bool leftover = false;
    module.walk([&](mlir::Operation *op) {
      if (is_unselected_source(op)) {
        op->emitError(
            "select-instructions: no pattern selected this operation");
        leftover = true;
      }
    });
    if (leftover) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::conversion::select_instructions
