#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "NormalizeLoopsPass.hpp"

namespace vesyla::transformation::normalize_loops {
#define GEN_PASS_DEF_NORMALIZELOOPSPASS
#include "transformation/normalize_loops/Passes.hpp.inc"

namespace {

bool normal(mlir::affine::AffineForOp loop) {
  return loop.hasConstantLowerBound() && loop.getConstantLowerBound() == 0 &&
         loop.getStepAsInt() == 1;
}

class NormalizeLoopsPass
    : public impl::NormalizeLoopsPassBase<NormalizeLoopsPass> {
public:
  using impl::NormalizeLoopsPassBase<
      NormalizeLoopsPass>::NormalizeLoopsPassBase;

  void runOnOperation() final {
    llvm::SmallVector<mlir::affine::AffineForOp> loops;
    getOperation().walk([&](mlir::affine::AffineForOp loop) {
      if (!normal(loop))
        loops.push_back(loop);
    });
    if (loops.empty())
      return;

    // normalizeAffineFor maps the counter back with an affine.apply; its
    // users take that into their own maps.
    llvm::SmallVector<mlir::Operation *> users;
    for (auto loop : loops) {
      if (mlir::failed(mlir::affine::normalizeAffineFor(loop))) {
        loop.emitError("normalize-loops: cannot rewrite this loop to start "
                       "at 0 with step 1");
        return signalPassFailure();
      }
      for (mlir::Operation *user : loop.getInductionVar().getUsers())
        if (llvm::isa<mlir::affine::AffineApplyOp>(user))
          for (mlir::Operation *u : user->getUsers())
            users.push_back(u);
    }

    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::affine::AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::affine::AffineLoadOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::affine::AffineStoreOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::affine::AffineVectorLoadOp::getCanonicalizationPatterns(patterns,
                                                                  ctx);
    mlir::affine::AffineVectorStoreOp::getCanonicalizationPatterns(patterns,
                                                                   ctx);
    (void)mlir::applyOpPatternsGreedily(users, std::move(patterns));
  }
};

} // namespace
} // namespace vesyla::transformation::normalize_loops
