#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "FlattenCellsPass.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_FLATTENCELLSPASS
#include "pasm/Passes.hpp.inc"

namespace {

class FlattenCellsPassRewriter : public mlir::OpRewritePattern<EpochOp> {
public:
  using mlir::OpRewritePattern<EpochOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EpochOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::Block &epoch_block = op.getBody().front();

    llvm::SmallVector<CellOp> cells;
    for (CellOp cell : epoch_block.getOps<CellOp>()) {
      cells.push_back(cell);
    }
    if (cells.empty()) {
      return mlir::failure();
    }

    for (CellOp cell : cells) {
      mlir::Block &cell_block = cell.getBody().front();

      llvm::SmallVector<mlir::Operation *> ops;
      for (mlir::Operation &inner : cell_block.without_terminator()) {
        ops.push_back(&inner);
      }
      for (mlir::Operation *inner : ops) {
        inner->moveBefore(&epoch_block, epoch_block.getTerminator()->getIterator());
      }

      rewriter.eraseOp(cell);
    }

    return mlir::success();
  }
};

class FlattenCellsPass : public impl::FlattenCellsPassBase<FlattenCellsPass> {
public:
  using impl::FlattenCellsPassBase<FlattenCellsPass>::FlattenCellsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FlattenCellsPassRewriter>(&getContext());
    mlir::FrozenRewritePatternSet pattern_set(std::move(patterns));
    if (failed(applyPatternsGreedily(module, pattern_set))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
