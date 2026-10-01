#include "NativeHelpers.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

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

// The address the access names outright, when it names a fixed one.
//
// An affine.load and its siblings carry their own map, whose first result is
// the bulk the access starts at -- the remaining results index within the bulk,
// which the AGU does not address. A constant there is the base; anything else
// is a value some loop supplies and belongs to the stepping instead.
static std::optional<int64_t> constant_base(mlir::Operation *access) {
  mlir::AffineMap map;
  if (auto read = llvm::dyn_cast<mlir::affine::AffineReadOpInterface>(access)) {
    map = read.getAffineMap();
  } else if (auto write =
                 llvm::dyn_cast<mlir::affine::AffineWriteOpInterface>(access)) {
    map = write.getAffineMap();
  } else {
    return std::nullopt;
  }
  if (map.getNumResults() == 0) {
    return std::nullopt;
  }
  auto constant = llvm::dyn_cast<mlir::AffineConstantExpr>(map.getResult(0));
  if (!constant) {
    return std::nullopt;
  }
  return constant.getValue();
}

mlir::AffineMapAttr lift_affine_map(mlir::Operation *access) {
  for (mlir::Value idx : access->getOperands()) {
    if (auto apply = idx.getDefiningOp<mlir::affine::AffineApplyOp>()) {
      return mlir::AffineMapAttr::get(apply.getAffineMap());
    }
  }

  mlir::MLIRContext *ctx = access->getContext();
  unsigned depth = enclosing_loop_depth(access);
  std::optional<int64_t> base = constant_base(access);

  // No loop to step and no base to start from: there is no address to describe.
  if (depth == 0 && (!base || *base == 0)) {
    return {};
  }

  mlir::AffineExpr addr = mlir::getAffineConstantExpr(base.value_or(0), ctx);
  if (depth > 0) {
    addr = mlir::getAffineDimExpr(depth - 1, ctx) + addr;
  }
  return mlir::AffineMapAttr::get(
      mlir::AffineMap::get(depth, /*symbolCount=*/0, addr));
}

} // namespace sel
} // namespace vesyla
