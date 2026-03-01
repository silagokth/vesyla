#include "pasm/Ops.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace vesyla;

enum class PasmRopPresence {
  None,    // No pasm.rop operations found
  Present, // At least one pasm.rop operation found
  Empty    // The loop is empty
};

struct RepOp {
  int iter, step = 0, delay = 0;
};

struct AffineForWithPasmRopPattern
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override;
};
