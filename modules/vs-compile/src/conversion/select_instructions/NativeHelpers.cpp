#include "NativeHelpers.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

bool is_io_buffer(mlir::Value memref) {
  auto alloc =
      llvm::dyn_cast_or_null<mlir::memref::AllocOp>(memref.getDefiningOp());
  if (!alloc) {
    return false;
  }
  auto id = alloc->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  if (!id) {
    return false;
  }
  return id.getValue() == "input_buffer" || id.getValue() == "output_buffer";
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

// The access's own first map result, re-expressed over the enclosing loops
// (outermost first), when it reads loop counters and nothing else. Null for a
// constant subscript and for anything that is not a function of enclosing
// loop counters alone -- those keep the heuristic below.
static mlir::AffineMapAttr lift_own_map(mlir::Operation *access,
                                        unsigned depth) {
  mlir::AffineMap map;
  llvm::SmallVector<mlir::Value> operands;
  if (auto read = llvm::dyn_cast<mlir::affine::AffineReadOpInterface>(access)) {
    map = read.getAffineMap();
    operands = llvm::to_vector(read.getMapOperands());
  } else if (auto write =
                 llvm::dyn_cast<mlir::affine::AffineWriteOpInterface>(access)) {
    map = write.getAffineMap();
    operands = llvm::to_vector(write.getMapOperands());
  } else {
    return {};
  }
  if (map.getNumResults() == 0 || map.getNumSymbols() != 0 ||
      llvm::isa<mlir::AffineConstantExpr>(map.getResult(0))) {
    return {};
  }

  llvm::SmallVector<mlir::Value> loops; // induction variables, outermost first
  for (mlir::Operation *parent = access->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      loops.insert(loops.begin(), loop.getInductionVar());
    }
  }

  mlir::MLIRContext *ctx = access->getContext();
  llvm::SmallVector<mlir::AffineExpr> dims;
  for (mlir::Value operand : operands) {
    auto at = llvm::find(loops, operand);
    if (at == loops.end()) {
      return {};
    }
    dims.push_back(mlir::getAffineDimExpr(at - loops.begin(), ctx));
  }
  mlir::AffineExpr addr = mlir::simplifyAffineExpr(
      map.getResult(0).replaceDims(dims), depth, /*numSymbols=*/0);
  return mlir::AffineMapAttr::get(
      mlir::AffineMap::get(depth, /*symbolCount=*/0, addr));
}

mlir::AffineMapAttr lift_affine_map(mlir::Operation *access) {
  for (mlir::Value idx : access->getOperands()) {
    if (auto apply = idx.getDefiningOp<mlir::affine::AffineApplyOp>()) {
      return mlir::AffineMapAttr::get(apply.getAffineMap());
    }
  }

  mlir::MLIRContext *ctx = access->getContext();
  unsigned depth = enclosing_loop_depth(access);

  // A subscript written over loop counters in the access's own map (what a
  // kernel lowered from C has: `%rf[%i]` inside a `%j` loop, `[%r + 2, 0]`):
  // that expression, over the enclosing loops outermost first, which is the
  // dim order the AGU reads a map in.
  if (auto lifted = lift_own_map(access, depth)) {
    return lifted;
  }

  std::optional<int64_t> base = constant_base(access);

  // No loop to step and no base to start from: there is no address to describe.
  if (depth == 0 && (!base || *base == 0)) {
    return {};
  }

  // A constant subscript names the same address on every iteration: the map
  // keeps the enclosing loops' dims, so each loop still gets its rep, but none
  // of them moves the address. Only a subscript some loop supplies is stepped,
  // and that by the innermost loop.
  mlir::AffineExpr addr = mlir::getAffineConstantExpr(base.value_or(0), ctx);
  if (depth > 0 && !base) {
    addr = mlir::getAffineDimExpr(depth - 1, ctx) + addr;
  }
  return mlir::AffineMapAttr::get(
      mlir::AffineMap::get(depth, /*symbolCount=*/0, addr));
}

} // namespace sel
} // namespace vesyla
