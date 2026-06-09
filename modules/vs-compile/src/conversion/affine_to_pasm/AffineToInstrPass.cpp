#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "AffineToInstrPass.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Support/Common.hpp"

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

// Lower a single rop that sits directly inside `loop`: append a `rep` instr
// derived from the loop's bounds and the rop's innermost map dim, then shrink
// the map by dropping that dim. Does not move the rop or touch the loop.
mlir::LogicalResult rewrite_rop_for_loop(pasm::RopOp rop,
                                         mlir::affine::AffineForOp loop,
                                         mlir::PatternRewriter &rewriter) {
  if (!rop.getMapAttr()) {
    return mlir::failure();
  }

  // get the coefficient of the current dim and use it as a step
  mlir::AffineMap map = rop.getMapAttr().getValue();
  if (map.getNumDims() == 0) {
    return mlir::failure();
  }
  unsigned target_dim = map.getNumDims() - 1;
  std::optional<int64_t> step = coefficient_of_dim(map.getResult(0), target_dim);
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
  mlir::AffineMap new_map =
      mlir::AffineMap::get(map.getNumDims() - 1, map.getNumSymbols(), new_expr);
  rop.setMapAttr(mlir::AffineMapAttr::get(new_map));

  return mlir::success();
}

// Lower an innermost affine.for whose body contains only rops. Every rop is
// lowered against this loop, then all rops are hoisted out (preserving order)
// and the now-empty loop is erased. A loop that still holds a nested affine.for
// is left untouched and only becomes innermost on a later greedy sweep, once
// its inner loops have been erased.
class RopLoopRewriter
    : public mlir::OpRewritePattern<mlir::affine::AffineForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::affine::AffineForOp loop,
                  mlir::PatternRewriter &rewriter) const final {
    // only trigger when every op in the loop body is a rop (no nested loops)
    llvm::SmallVector<pasm::RopOp, 4> rops;
    for (mlir::Operation &op : loop.getBody()->without_terminator()) {
      auto rop = llvm::dyn_cast<pasm::RopOp>(op);
      if (!rop) {
        return mlir::failure();
      }
      rops.push_back(rop);
    }
    if (rops.empty()) {
      return mlir::failure();
    }

    // lower each rop against this loop
    for (pasm::RopOp rop : rops) {
      if (mlir::failed(rewrite_rop_for_loop(rop, loop, rewriter))) {
        return mlir::failure();
      }
    }

    // hoist all rops out before the loop, preserving order, then erase it
    for (pasm::RopOp rop : rops) {
      rop->moveBefore(loop);
    }
    rewriter.eraseOp(loop);

    return mlir::success();
  }
};

class AffineToInstrPass
    : public impl::AffineToInstrPassBase<AffineToInstrPass> {
public:
  using impl::AffineToInstrPassBase<AffineToInstrPass>::AffineToInstrPassBase;

  // This pass works on the assumption that each affine.for body holds only rops
  // and nested affine.for ops, and that the depth of nesting enclosing a rop is
  // equal to the dimension of its affine map. A loop may hold several rops; each
  // is lowered against the loop and hoisted out. It fails if these conditions do
  // not hold.
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
          if (!llvm::isa<pasm::RopOp>(op) &&
              !llvm::isa<mlir::affine::AffineForOp>(op)) {
            rop.emitError("affine.for parent contains an op that is neither a "
                          "pasm.rop nor a nested affine.for");
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
    patterns.add<RopLoopRewriter>(&getContext());
    if (mlir::failed(applyPatternsGreedily(
            getOperation(),
            mlir::FrozenRewritePatternSet(std::move(patterns))))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::conversion::affine_to_pasm
