#include "vesyla/Support/LoopShape.hpp"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include <functional>

namespace vesyla::util {

namespace {

// The coefficient of dim `pos` in an affine expression of sums and products
// with constants; nothing for anything else.
std::optional<int64_t> coefficient(mlir::AffineExpr e, unsigned pos) {
  if (llvm::isa<mlir::AffineConstantExpr>(e) ||
      llvm::isa<mlir::AffineSymbolExpr>(e)) {
    return 0;
  }
  if (auto d = llvm::dyn_cast<mlir::AffineDimExpr>(e)) {
    return d.getPosition() == pos ? 1 : 0;
  }
  auto bin = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(e);
  if (!bin) {
    return std::nullopt;
  }
  auto l = coefficient(bin.getLHS(), pos);
  auto r = coefficient(bin.getRHS(), pos);
  if (!l || !r) {
    return std::nullopt;
  }
  switch (bin.getKind()) {
  case mlir::AffineExprKind::Add:
    return *l + *r;
  case mlir::AffineExprKind::Mul:
    if (auto c = llvm::dyn_cast<mlir::AffineConstantExpr>(bin.getRHS())) {
      return *l * c.getValue();
    }
    if (auto c = llvm::dyn_cast<mlir::AffineConstantExpr>(bin.getLHS())) {
      return c.getValue() * *r;
    }
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

// The constant term of such an expression.
std::optional<int64_t> constant(mlir::AffineExpr e) {
  if (auto c = llvm::dyn_cast<mlir::AffineConstantExpr>(e)) {
    return c.getValue();
  }
  if (llvm::isa<mlir::AffineDimExpr>(e)) {
    return 0;
  }
  auto bin = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(e);
  if (!bin) {
    return std::nullopt;
  }
  auto l = constant(bin.getLHS());
  auto r = constant(bin.getRHS());
  if (!l || !r) {
    return std::nullopt;
  }
  switch (bin.getKind()) {
  case mlir::AffineExprKind::Add:
    return *l + *r;
  case mlir::AffineExprKind::Mul:
    return *l * *r;
  default:
    return std::nullopt;
  }
}

} // namespace

llvm::SmallVector<mlir::affine::AffineForOp>
enclosing_loops(mlir::Operation *op) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      loops.insert(loops.begin(), loop);
    }
  }
  return loops;
}

std::optional<LoopShape> loop_shape(mlir::affine::AffineForOp loop) {
  if (loop.getStepAsInt() != 1 || !loop.hasConstantLowerBound() ||
      loop.getConstantLowerBound() != 0) {
    return std::nullopt;
  }
  if (loop.hasConstantUpperBound()) {
    return LoopShape{loop.getConstantUpperBound(), 0, 0};
  }
  mlir::AffineMap ub = loop.getUpperBoundMap();
  if (ub.getNumResults() != 1 || ub.getNumSymbols() != 0) {
    return std::nullopt;
  }
  mlir::AffineExpr e = ub.getResult(0);
  std::optional<int64_t> k = constant(e);
  if (!k) {
    return std::nullopt;
  }
  auto operands = loop.getUpperBoundOperands();
  std::optional<unsigned> used;
  int64_t slope = 0;
  for (unsigned pos = 0; pos < ub.getNumDims(); ++pos) {
    std::optional<int64_t> c = coefficient(e, pos);
    if (!c) {
      return std::nullopt;
    }
    if (*c == 0) {
      continue;
    }
    if (used) {
      return std::nullopt;
    }
    used = pos;
    slope = *c;
  }
  if (!used) {
    return LoopShape{*k, 0, 0};
  }
  mlir::affine::AffineForOp followed =
      mlir::affine::getForInductionVarOwner(operands[*used]);
  if (!followed) {
    return std::nullopt;
  }
  unsigned distance = 0;
  for (mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto outer = llvm::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      ++distance;
      if (outer == followed) {
        return LoopShape{*k, slope, distance};
      }
    }
  }
  return std::nullopt;
}

mlir::affine::AffineForOp followed_loop(mlir::affine::AffineForOp loop) {
  std::optional<LoopShape> shape = loop_shape(loop);
  if (!shape || shape->constant()) {
    return nullptr;
  }
  unsigned distance = 0;
  for (mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto outer = llvm::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      if (++distance == shape->follows) {
        return outer;
      }
    }
  }
  return nullptr;
}

int64_t trips_at(mlir::affine::AffineForOp loop, llvm::ArrayRef<int64_t> idx) {
  std::optional<LoopShape> shape = loop_shape(loop);
  if (!shape) {
    return 1;
  }
  if (shape->constant()) {
    return shape->iter;
  }
  if (shape->follows > idx.size()) {
    return shape->iter;
  }
  return shape->iter + shape->slope * idx[idx.size() - shape->follows];
}

int64_t max_trips(mlir::affine::AffineForOp loop) {
  std::optional<LoopShape> shape = loop_shape(loop);
  if (!shape) {
    return 1;
  }
  if (shape->constant() || shape->slope <= 0) {
    return shape->iter;
  }
  mlir::affine::AffineForOp followed = followed_loop(loop);
  return shape->iter + shape->slope * (max_trips(followed) - 1);
}

int64_t trips_at_last(mlir::affine::AffineForOp loop) {
  std::optional<LoopShape> shape = loop_shape(loop);
  if (!shape) {
    return 1;
  }
  if (shape->constant()) {
    return shape->iter;
  }
  return shape->iter + shape->slope * (trips_at_last(followed_loop(loop)) - 1);
}

llvm::SmallVector<int64_t>
last_indices(llvm::ArrayRef<mlir::affine::AffineForOp> loops) {
  llvm::SmallVector<int64_t> idx;
  for (mlir::affine::AffineForOp loop : loops) {
    idx.push_back(trips_at_last(loop) - 1);
  }
  return idx;
}

bool for_each_point(llvm::ArrayRef<mlir::affine::AffineForOp> loops,
                    int64_t limit,
                    llvm::function_ref<void(llvm::ArrayRef<int64_t>)> visit) {
  for (mlir::affine::AffineForOp loop : loops) {
    if (!loop_shape(loop)) {
      return false;
    }
  }
  llvm::SmallVector<int64_t> idx(loops.size(), 0);
  int64_t points = 0;
  std::function<bool(size_t)> walk = [&](size_t depth) -> bool {
    if (depth == loops.size()) {
      if (++points > limit) {
        return false;
      }
      visit(idx);
      return true;
    }
    int64_t trips =
        trips_at(loops[depth], llvm::ArrayRef<int64_t>(idx).take_front(depth));
    for (int64_t i = 0; i < trips; ++i) {
      idx[depth] = i;
      if (!walk(depth + 1)) {
        return false;
      }
    }
    return true;
  };
  return walk(0);
}

} // namespace vesyla::util
