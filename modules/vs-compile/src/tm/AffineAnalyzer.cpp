#include "tm/AffineAnalyzer.hpp"
#include "tm/AffineForWithPasmRopPattern.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::affine::AffineForOp;

void processAffineForOp(AffineForOp forOp, int level = 0) {
  std::string indent(level * 2, ' ');
  llvm::errs() << indent << "affine.for\n";
  for (auto &op : forOp.getBody()->getOperations()) {
    if (auto nestedForOp = llvm::dyn_cast<AffineForOp>(&op)) {
      processAffineForOp(nestedForOp, level + 1);
    } else {
      llvm::errs() << indent << "  " << op.getName() << "\n";
    }
  }
}

void AffineAnalyzerPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<AffineForWithPasmRopPattern>(&getContext());

  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
