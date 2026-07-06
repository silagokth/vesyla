#include "NativeHelpers.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"

namespace vesyla {
namespace sel {

namespace {

// Number of affine.for ops enclosing `op`.
unsigned enclosing_loop_depth(mlir::Operation *op) {
  unsigned depth = 0;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<mlir::affine::AffineForOp>(parent)) {
      ++depth;
    }
  }
  return depth;
}

// Best-effort lift of the affine access map onto the rop's `map` attribute.
//   - outside any affine.for: no map (returns null; the access is not iterated).
//   - inside affine.for(s): reuse the map of the affine.apply that feeds an
//     index operand, if any; otherwise fall back to identity onto the innermost
//     loop dim.
// NOTE: this is a heuristic that matches the shapes in the reference IR; the
// exact multi-index / symbol cases will need revisiting once run end-to-end.
mlir::AffineMapAttr lift_affine_map(mlir::Operation *access) {
  unsigned depth = enclosing_loop_depth(access);
  if (depth == 0) {
    return {};
  }
  for (mlir::Value idx : access->getOperands()) {
    if (auto apply = idx.getDefiningOp<mlir::affine::AffineApplyOp>()) {
      return mlir::AffineMapAttr::get(apply.getAffineMap());
    }
  }
  mlir::MLIRContext *ctx = access->getContext();
  return mlir::AffineMapAttr::get(
      mlir::AffineMap::get(depth, /*symbolCount=*/0,
                           mlir::getAffineDimExpr(depth - 1, ctx)));
}

// Build a drra.rop. RopOp declares no named attributes, so it is built via an
// OperationState with the attribute dictionary attached directly.
mlir::Operation *build_rop(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::TypeRange results, mlir::ValueRange operands,
                           llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::OperationState state(loc, drra::RopOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attrs);
  return rewriter.create(state);
}

// Common attributes shared by every emitted rop, copying `id`/`resource` from
// the source op when present.
llvm::SmallVector<mlir::NamedAttribute>
common_attrs(mlir::Builder &b, mlir::Operation *src, llvm::StringRef kind,
             llvm::StringRef instr) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  if (mlir::Attribute id = src->getAttr("id")) {
    attrs.push_back(b.getNamedAttr("id", id));
  }
  attrs.push_back(b.getNamedAttr("kind", b.getStringAttr(kind)));
  attrs.push_back(b.getNamedAttr("instr", b.getStringAttr(instr)));
  if (mlir::Attribute resource = src->getAttr("resource")) {
    attrs.push_back(b.getNamedAttr("resource", resource));
  }
  return attrs;
}

} // namespace

bool is_iter_args_carry(mlir::Value acc) {
  auto arg = llvm::dyn_cast<mlir::BlockArgument>(acc);
  if (!arg) {
    return false;
  }
  auto loop =
      llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(arg.getOwner()->getParentOp());
  if (!loop) {
    return false;
  }
  // Argument 0 is the induction variable; the iter_args carries start at 1.
  return arg.getArgNumber() >= 1;
}

void emit_mac(mlir::PatternRewriter &rewriter, mlir::Operation *mul,
              mlir::Operation *add, mlir::Value a, mlir::Value b) {
  mlir::Builder b_(rewriter.getContext());
  // The MAC takes its id/resource from the multiply (dest,src,src resources).
  llvm::SmallVector<mlir::NamedAttribute> attrs =
      common_attrs(b_, mul, /*kind=*/"dpu", /*instr=*/"conf");
  // mode=10 is the ISA encoding of "mac" on the DPU (verbo_map mode 10 -> mac).
  attrs.push_back(b_.getNamedAttr(
      "param", b_.getDictionaryAttr(
                   {b_.getNamedAttr("mode", b_.getI32IntegerAttr(10))})));
  // A compute op does not index memory: its map is constant 0 over the enclosing
  // loop dims, so AffineToRepPass derives a step of 0 (it repeats each
  // iteration rather than sweeping an address).
  if (unsigned depth = enclosing_loop_depth(mul)) {
    attrs.push_back(b_.getNamedAttr(
        "map", mlir::AffineMapAttr::get(mlir::AffineMap::get(
                   depth, /*symbolCount=*/0,
                   mlir::getAffineConstantExpr(0, rewriter.getContext())))));
  }

  mlir::Operation *rop = build_rop(rewriter, add->getLoc(),
                                   add->getResultTypes(), {a, b}, attrs);
  rewriter.replaceOp(add, rop->getResults());
  // `mul` fed only `add`; it is now dead.
  rewriter.eraseOp(mul);
}

void emit_rf_load(mlir::PatternRewriter &rewriter, mlir::Operation *load) {
  mlir::Builder b_(rewriter.getContext());
  llvm::SmallVector<mlir::NamedAttribute> attrs =
      common_attrs(b_, load, /*kind=*/"rf", /*instr=*/"evt");
  attrs.push_back(b_.getNamedAttr(
      "param", b_.getDictionaryAttr(
                   {b_.getNamedAttr("init_addr", b_.getI32IntegerAttr(0))})));
  if (mlir::AffineMapAttr map = lift_affine_map(load)) {
    attrs.push_back(b_.getNamedAttr("map", map));
  }

  mlir::Operation *rop = build_rop(rewriter, load->getLoc(),
                                   load->getResultTypes(), /*operands=*/{}, attrs);
  rewriter.replaceOp(load, rop->getResults());
}

void emit_rf_store(mlir::PatternRewriter &rewriter, mlir::Operation *store) {
  mlir::Builder b_(rewriter.getContext());
  llvm::SmallVector<mlir::NamedAttribute> attrs =
      common_attrs(b_, store, /*kind=*/"rf", /*instr=*/"evt");
  attrs.push_back(b_.getNamedAttr(
      "param", b_.getDictionaryAttr(
                   {b_.getNamedAttr("init_addr", b_.getI32IntegerAttr(0))})));
  if (mlir::AffineMapAttr map = lift_affine_map(store)) {
    attrs.push_back(b_.getNamedAttr("map", map));
  }

  // For affine.store / affine.vector_store the value being stored is operand 0.
  mlir::Value stored = store->getOperand(0);
  build_rop(rewriter, store->getLoc(), /*results=*/{}, {stored}, attrs);
  rewriter.eraseOp(store);
}

} // namespace sel
} // namespace vesyla
