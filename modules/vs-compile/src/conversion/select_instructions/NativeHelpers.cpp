#include "NativeHelpers.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"

namespace vesyla {
namespace sel {

bool is_iter_args_carry(mlir::Value acc) {
  auto arg = llvm::dyn_cast<mlir::BlockArgument>(acc);
  if (!arg) {
    return false;
  }
  auto loop = llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(
      arg.getOwner()->getParentOp());
  if (!loop) {
    return false;
  }
  // Argument 0 is the induction variable; the iter_args carries start at 1.
  return arg.getArgNumber() >= 1;
}

bool feeds_iter_args_init(mlir::Operation *op) {
  for (mlir::Value result : op->getResults()) {
    for (mlir::Operation *user : result.getUsers()) {
      auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(user);
      if (loop && llvm::is_contained(loop.getInits(), result)) {
        return true;
      }
    }
  }
  return false;
}

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

} // namespace sel
} // namespace vesyla
