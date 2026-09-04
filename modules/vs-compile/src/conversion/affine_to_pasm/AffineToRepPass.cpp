#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include "AffineToRepPass.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Support/Common.hpp"
#include "vesyla/Support/Config.hpp"

namespace vesyla::conversion::affine_to_pasm {
#define GEN_PASS_DEF_AFFINETOREPPASS
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

// The part of an address expression that no loop moves: the address the AGU
// starts its sweep from.
//
// The coefficient of each dim becomes that loop's rep step, so what is left
// once every dim is taken as zero is the base. Returns nothing for an
// expression this cannot evaluate -- a symbol, a division, a modulo -- rather
// than guessing at a base address.
std::optional<int64_t> constant_term(mlir::AffineExpr expr) {
  using mlir::AffineBinaryOpExpr;
  using mlir::AffineConstantExpr;
  using mlir::AffineDimExpr;
  using mlir::AffineExprKind;

  if (auto constant = llvm::dyn_cast<AffineConstantExpr>(expr)) {
    return constant.getValue();
  }
  if (llvm::isa<AffineDimExpr>(expr)) {
    return 0;
  }
  auto bin = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin) {
    return std::nullopt;
  }
  std::optional<int64_t> lhs = constant_term(bin.getLHS());
  std::optional<int64_t> rhs = constant_term(bin.getRHS());
  if (!lhs || !rhs) {
    return std::nullopt;
  }
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return *lhs + *rhs;
  case AffineExprKind::Mul:
    return *lhs * *rhs;
  default:
    return std::nullopt;
  }
}

// An instruction takes a base address iff its ISA definition carries an
// "init_addr" segment. That is a property of the instruction name and holds
// across the components defining it, so it is looked up by name -- the same
// question AddSlotPortPass asks of "port", asked the same way.
bool instr_takes_init_addr(const nlohmann::json &isa_json,
                           llvm::StringRef instr_name) {
  auto has_init_addr = [](const nlohmann::json &segments) {
    for (const auto &segment : segments) {
      if (segment.contains("name") && segment["name"] == "init_addr") {
        return true;
      }
    }
    return false;
  };
  for (const auto &component : isa_json["components"]) {
    for (const auto &instr : component["instructions"]) {
      if (!instr.contains("name") || instr["name"] != instr_name.str()) {
        continue;
      }
      if (instr.contains("segments") && has_init_addr(instr["segments"])) {
        return true;
      }
      if (instr.contains("variants")) {
        for (const auto &variant : instr["variants"]) {
          if (variant.contains("segments") && has_init_addr(variant["segments"])) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Write the base address of the rop's map onto the instruction that takes one.
//
// The loop-dependent part of the map becomes the rep steps below; the constant
// term belongs to no loop and is where the sweep starts. Without it
// `(d0) -> (d0 + 1)` and `(d0) -> (d0)` lower to the same instructions and the
// access runs one address low. A rop whose instruction has no such segment --
// a dpu conf, a dpu event -- is left alone, and so is a base of zero, which is
// what the default already is.
mlir::LogicalResult set_init_addr(pasm::RopOp rop,
                                  const nlohmann::json &isa_json) {
  if (!rop.getMapAttr()) {
    return mlir::success();
  }
  mlir::AffineMap map = rop.getMapAttr().getValue();
  if (map.getNumResults() != 1) {
    return mlir::success();
  }
  std::optional<int64_t> base = constant_term(map.getResult(0));
  if (!base) {
    rop.emitError("affine map has a base address this pass cannot evaluate");
    return mlir::failure();
  }
  if (*base == 0) {
    return mlir::success();
  }

  mlir::OpBuilder builder(rop.getContext());
  for (mlir::Operation &op : rop.getBody().front()) {
    auto instr = llvm::dyn_cast<pasm::InstrOp>(op);
    if (!instr || !instr_takes_init_addr(isa_json, instr.getType())) {
      continue;
    }
    if (instr.getParam().get("init_addr")) {
      continue;
    }
    llvm::SmallVector<mlir::NamedAttribute> params(instr.getParam().begin(),
                                                   instr.getParam().end());
    params.push_back(builder.getNamedAttr(
        "init_addr", builder.getI32IntegerAttr(static_cast<int32_t>(*base))));
    instr->setAttr("param", builder.getDictionaryAttr(params));
  }
  return mlir::success();
}

// Number of affine.for ops enclosing `op`.
unsigned enclosing_loop_count(mlir::Operation *op) {
  unsigned count = 0;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<mlir::affine::AffineForOp>(parent)) {
      ++count;
    }
  }
  return count;
}

// Lower a single rop that sits directly inside `loop`: append a `rep` instr
// derived from the loop's bounds and the rop's innermost map dim. Reads a local
// copy of the map and leaves the rop's stored map untouched. Does not move the
// rop or touch the loop.
mlir::LogicalResult rewrite_rop_for_loop(pasm::RopOp rop,
                                         mlir::affine::AffineForOp loop,
                                         mlir::PatternRewriter &rewriter) {
  if (!rop.getMapAttr()) {
    return mlir::failure();
  }

  // Read a local copy of the map; never write it back. The dim consumed for
  // this loop is the innermost still-enclosing one (depth - 1), derived from
  // the current nesting rather than from a shrinking map, so the original map
  // is preserved across sweeps.
  mlir::AffineMap map = rop.getMapAttr().getValue();
  unsigned depth = enclosing_loop_count(rop);
  if (depth == 0 || depth > map.getNumDims()) {
    return mlir::failure();
  }
  unsigned target_dim = depth - 1;
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

  // The rep acts on the same port as the rop's resource. The port is carried on
  // the existing instruction(s) in the rop body (the rop no longer holds it),
  // so copy it across to the rep.
  int32_t port = 0;
  for (mlir::Operation &sibling : rop_body) {
    if (auto instr = llvm::dyn_cast<pasm::InstrOp>(sibling)) {
      if (auto port_attr =
              llvm::dyn_cast_or_null<mlir::IntegerAttr>(instr.getParam().get("port"))) {
        port = static_cast<int32_t>(port_attr.getInt());
        break;
      }
    }
  }

  // insert the rep instructions
  auto params = rewriter.getDictionaryAttr({
      rewriter.getNamedAttr("delay", rewriter.getStringAttr(delay_value)),
      rewriter.getNamedAttr(
          "iter", rewriter.getI32IntegerAttr(static_cast<int32_t>(iter))),
      rewriter.getNamedAttr(
          "step", rewriter.getI32IntegerAttr(static_cast<int32_t>(*step))),
      rewriter.getNamedAttr("port", rewriter.getI32IntegerAttr(port)),
  });
  pasm::InstrOp::create(rewriter, loop.getLoc(),
                        rewriter.getStringAttr(instr_id),
                        rewriter.getStringAttr("rep"), params);

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

class AffineToRepPass
    : public impl::AffineToRepPassBase<AffineToRepPass> {
public:
  using impl::AffineToRepPassBase<AffineToRepPass>::AffineToRepPassBase;

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

    // The base address first, while every map is still on its rop: the rewriter
    // below consumes the loops the map's dims stand for.
    vesyla::pasm::Config cfg;
    nlohmann::json isa_json = cfg.get_isa_json();
    bool addressed = true;
    getOperation().walk([&](pasm::RopOp rop) {
      if (mlir::failed(set_init_addr(rop, isa_json))) {
        addressed = false;
      }
    });
    if (!addressed) {
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
