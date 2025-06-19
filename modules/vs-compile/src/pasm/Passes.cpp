#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "pasm/Passes.hpp"
#include "tm/Solver.hpp"
#include "tm/TimingModel.hpp"
#include "util/Common.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_REPLACELOOPOP
#include "pasm/Passes.hpp.inc"

std::string gen_random_string(size_t length) {
  if (length == 0) {
    return "";
  }

  // generate a random string starting with a alphabetic character
  const std::string letters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const std::string numbers = "0123456789";
  const std::string characters = letters + numbers;
  std::string result;
  result.reserve(length);
  result += letters[rand() % letters.size()];
  for (size_t i = 1; i < length; ++i) {
    result += characters[rand() % characters.size()];
  }
  return result;
}

namespace {

//===----------------------------------------------------------------------===//
class ReplaceLoopOpRewriter : public OpRewritePattern<LoopOp> {
public:
  using OpRewritePattern<LoopOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter &rewriter) const final {

    int iter = op.getIter();
    auto &body = op.getBody();
    // return failure();
    // Get the block containing the LoopOp itself
    mlir::Block *parentBlock = op.getOperation()->getBlock();

    if (parentBlock) {
      llvm::outs() << "LoopOp is in block: " << parentBlock << "\n";
      std::string originalIdStr = op.getId().str();
      std::string newId = "epoch_derived_from_loop_" + originalIdStr;
      mlir::StringAttr newIdAttr = rewriter.getStringAttr(newId);
      EpochOp epoch_op = rewriter.create<EpochOp>(op.getLoc(), newIdAttr);
      mlir::Region &epochBodyRegion = epoch_op.getBody();
      mlir::Block *entryBlock;
      if (epochBodyRegion.empty()) {
        entryBlock = rewriter.createBlock(&epochBodyRegion);
      } else {
        entryBlock = &epochBodyRegion.front();
      }
      rewriter.setInsertionPointToEnd(entryBlock);
      // Move the operations from the original loop body's blocks
      // into the new epoch's entry block, before the terminator.
      for (mlir::Block &block :
           body) { // Iterate through blocks in the LoopOp's region
        for (mlir::Operation &child_op :
             block) { // Iterate ops excluding the terminator
          // Clone the operation at the current insertion point
          rewriter.clone(child_op);
        }
        // Note: This simple loop assumes a single block in the LoopOp
        // body and doesn't handle block arguments or complex control flow
        // transfer. If LoopOp body can have multiple blocks or branches,
        // a more sophisticated merging/cloning logic (like
        // inlineRegionBefore) is needed.
      }
      rewriter.replaceOp(op, epoch_op);

    } else {
      // This usually means the op is top-level (like a FuncOp)
      llvm::outs() << "LoopOp is not inside a block.\n";
    }

    // create a new
    return failure();
  }
};

class ReplaceLoopOp : public impl::ReplaceLoopOpBase<ReplaceLoopOp> {
public:
  using impl::ReplaceLoopOpBase<ReplaceLoopOp>::ReplaceLoopOpBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceLoopOpRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace vesyla::pasm