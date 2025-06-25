#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "AddHaltPass.hpp"
#include "util/RandName.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_ADDHALTPASS
#include "pasm/Passes.hpp.inc"

namespace {
//===----------------------------------------------------------------------===//
class AddHaltPassRewriter : public OpRewritePattern<RawOp> {
public:
  using OpRewritePattern<RawOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(RawOp op,
                                PatternRewriter &rewriter) const final {

    mlir::Region &op_region = op.getBody();
    mlir::Block *op_block = nullptr;
    if (op_region.empty()) {
      op_block = rewriter.createBlock(&op_region);
    } else {
      op_block = &op_region.front();
    }

    // check if the second last operation is a halt instruction
    if (op_block->getOperations().size() > 1) {
      auto second_last_op = op_block->getOperations().rbegin();
      std::advance(second_last_op, 1);
      if (auto instr_op =
              llvm::dyn_cast<vesyla::pasm::InstrOp>(*second_last_op)) {
        if (instr_op.getType() == "halt") {
          return mlir::failure();
        }
      }
    }

    rewriter.setInsertionPoint(op_block->getTerminator());
    // Create a new Halt instruction
    auto halt_op = rewriter.create<vesyla::pasm::InstrOp>(
        op.getLoc(), rewriter.getStringAttr(util::RandName::generate(8)),
        rewriter.getStringAttr("halt"), rewriter.getDictionaryAttr({}));

    return mlir::success();
  }
};

class AddHaltPass : public impl::AddHaltPassBase<AddHaltPass> {
public:
  using impl::AddHaltPassBase<AddHaltPass>::AddHaltPassBase;

  void runOnOperation() {
    // Get the current module
    mlir::ModuleOp module = getOperation();

    // Create a pattern set
    RewritePatternSet patterns(&getContext());
    patterns.add<AddHaltPassRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    // Apply the patterns to the module
    if (failed(applyPatternsGreedily(module, patternSet))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm