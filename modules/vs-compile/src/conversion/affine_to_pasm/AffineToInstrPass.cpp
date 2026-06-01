#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "AffineToInstrPass.hpp"
#include "pasm/Ops.hpp"
#include "util/Common.hpp"

namespace vesyla::conversion::affine_to_pasm {
#define GEN_PASS_DEF_AFFINETOINSTRPASS
#include "conversion/affine_to_pasm/Passes.hpp.inc"

namespace {

std::optional<int64_t> coefficient_of_dim(mlir::AffineExpr expr,
                                          unsigned target_dim) {
  using mlir::AffineBinaryOpExpr;
  using mlir::AffineConstantExpr;
  using mlir::AffineDimExpr;
  using mlir::AffineExprKind;
  using mlir::AffineSymbolExpr;

  if (llvm::isa<AffineConstantExpr>(expr)) {
    return 0;
  }
  if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
    return dim.getPosition() == target_dim ? 1 : 0;
  }
  if (llvm::isa<AffineSymbolExpr>(expr)) {
    return 0;
  }
  auto bin = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin) {
    return std::nullopt;
  }
  mlir::AffineExpr lhs = bin.getLHS();
  mlir::AffineExpr rhs = bin.getRHS();
  switch (bin.getKind()) {
  case AffineExprKind::Add: {
    auto l = coefficient_of_dim(lhs, target_dim);
    auto r = coefficient_of_dim(rhs, target_dim);
    if (!l || !r) {
      return std::nullopt;
    }
    return *l + *r;
  }
  case AffineExprKind::Mul: {
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(lhs)) {
      auto r = coefficient_of_dim(rhs, target_dim);
      if (!r) {
        return std::nullopt;
      }
      return c.getValue() * *r;
    }
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(rhs)) {
      auto l = coefficient_of_dim(lhs, target_dim);
      if (!l) {
        return std::nullopt;
      }
      return *l * c.getValue();
    }
    return std::nullopt;
  }
  default:
    return std::nullopt;
  }
}

class RopRewriter : public mlir::OpRewritePattern<pasm::RopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pasm::RopOp rop,
                  mlir::PatternRewriter &rewriter) const final {
    if (!rop.getMapAttr()) {
      return mlir::failure();
    }
    auto loop = llvm::dyn_cast_if_present<mlir::affine::AffineForOp>(
        rop->getParentOp());
    if (!loop) {
      return mlir::failure();
    }

    // get the coefficient of the current dim and use it as a step
    mlir::AffineMap map = rop.getMapAttr().getValue();
    if (map.getNumDims() == 0) {
      return mlir::failure();
    }
    unsigned target_dim = map.getNumDims() - 1;
    std::optional<int64_t> step =
        coefficient_of_dim(map.getResult(0), target_dim);
    if (!step) {
      rop.emitError("affine map contains operations other than add and mul");
      return mlir::failure();
    }
    // the iterations should be the same as the affine.for
    int64_t iter = loop.getConstantUpperBound();

    // use the location of the affine.for for debugging purposes
    mlir::StringAttr loc_name;
    if (auto nloc = llvm::dyn_cast<mlir::NameLoc>(loop.getLoc())) {
      loc_name = nloc.getName();
    } else {
      loc_name =
          rewriter.getStringAttr(vesyla::util::Common::gen_random_string(8));
    }

    // generate ids that help with debugging
    std::string instr_id = rop.getSymName().str() + "_" + loc_name.str();
    std::string delay_value = "t_" + loc_name.str();

    // get insertions point
    mlir::Block &rop_body = rop.getBody().front();
    mlir::Operation *yield = rop_body.getTerminator();
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yield);

    // insert the rep instructions
    auto params = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("delay", rewriter.getStringAttr(delay_value)),
        rewriter.getNamedAttr(
            "iter", rewriter.getI32IntegerAttr(static_cast<int32_t>(iter))),
        rewriter.getNamedAttr(
            "step", rewriter.getI32IntegerAttr(static_cast<int32_t>(*step))),
    });
    pasm::InstrOp::create(rewriter, loop.getLoc(),
                          rewriter.getStringAttr(instr_id),
                          rewriter.getStringAttr("rep"), params);

    // shrink the map by dropping the consumed innermost dim; any residual
    // constant stays in the result expression of the now-smaller map
    mlir::MLIRContext *ctx = map.getContext();
    llvm::SmallVector<mlir::AffineExpr, 4> dim_replacements;
    for (unsigned i = 0; i + 1 < map.getNumDims(); ++i) {
      dim_replacements.push_back(mlir::getAffineDimExpr(i, ctx));
    }
    dim_replacements.push_back(mlir::getAffineConstantExpr(0, ctx));
    mlir::AffineExpr new_expr = map.getResult(0).replaceDimsAndSymbols(
        dim_replacements, /*symReplacements=*/{});
    mlir::AffineMap new_map = mlir::AffineMap::get(
        map.getNumDims() - 1, map.getNumSymbols(), new_expr);
    rop.setMapAttr(mlir::AffineMapAttr::get(new_map));

    // delete the loop above and return
    rop->moveBefore(loop);
    rewriter.eraseOp(loop);

    return mlir::success();
  }
};

class AffineToInstrPass
    : public impl::AffineToInstrPassBase<AffineToInstrPass> {
public:
  using impl::AffineToInstrPassBase<AffineToInstrPass>::AffineToInstrPassBase;

  // This pass works on the assumption that there is only a single rop in each
  // nested affine.for and the depth of nesting is equal to the dimension of the
  // affine map of the rop, it fails if any of these conditions fail
  void runOnOperation() final {
    bool fatal = false;
    getOperation().walk([&](pasm::RopOp rop) {
      bool has_map = static_cast<bool>(rop.getMapAttr());
      auto loop = llvm::dyn_cast_if_present<mlir::affine::AffineForOp>(
          rop->getParentOp());
      bool has_loop_parent = static_cast<bool>(loop);
      if (has_map && !has_loop_parent &&
          rop.getMapAttr().getValue().getNumDims() > 0) {
        rop.emitError(
            "rop has an affine map with dims but no affine.for parent");
        fatal = true;
      }
      if (!has_map && has_loop_parent) {
        rop.emitError("rop is inside an affine.for but has no affine map");
        fatal = true;
      }
      if (has_loop_parent) {
        for (mlir::Operation &op : loop.getBody()->without_terminator()) {
          if (&op != rop.getOperation()) {
            rop.emitError("affine.for parent contains operations other than "
                          "this rop and the terminator");
            fatal = true;
            break;
          }
        }
        if (!loop.hasConstantLowerBound() ||
            loop.getConstantLowerBound() != 0) {
          rop.emitError("affine.for parent must have constant lower bound 0");
          fatal = true;
        }
        if (loop.getStepAsInt() != 1) {
          rop.emitError("affine.for parent must have step 1");
          fatal = true;
        }
        if (!loop.hasConstantUpperBound()) {
          rop.emitError("affine.for parent must have a constant upper bound");
          fatal = true;
        }
      }
    });
    if (fatal) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RopRewriter>(&getContext());
    if (mlir::failed(applyPatternsGreedily(
            getOperation(),
            mlir::FrozenRewritePatternSet(std::move(patterns))))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::conversion::affine_to_pasm
