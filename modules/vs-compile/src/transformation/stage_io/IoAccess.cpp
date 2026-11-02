#include "IoAccess.hpp"

#include "conversion/select_instructions/NativeHelpers.hpp"

namespace vesyla::transformation::stage_io {

std::optional<IoAccess> as_io_access(mlir::Operation *op) {
  IoAccess access;
  if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
    access = {op, false, load.getMemRef(), load.getAffineMap(),
              llvm::to_vector(load.getMapOperands())};
  } else if (auto store = llvm::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
    access = {op, true, store.getMemRef(), store.getAffineMap(),
              llvm::to_vector(store.getMapOperands())};
  } else {
    return std::nullopt;
  }
  if (!vesyla::sel::is_io_buffer(access.buffer) ||
      access.map.getNumResults() != 2) {
    return std::nullopt;
  }
  return access;
}

llvm::SmallVector<IoAccess> collect_io_accesses(mlir::Operation *scope) {
  llvm::SmallVector<IoAccess> accesses;
  scope->walk([&](mlir::Operation *op) {
    if (auto access = as_io_access(op)) {
      accesses.push_back(*access);
    }
  });
  return accesses;
}

llvm::SmallVector<mlir::affine::AffineForOp> loops_of(const IoAccess &access,
                                                      mlir::AffineExpr e) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops;
  for (auto [pos, operand] : llvm::enumerate(access.operands)) {
    if (pos < access.map.getNumDims() && e.isFunctionOfDim(pos)) {
      loops.push_back(mlir::affine::getForInductionVarOwner(operand));
    }
  }
  return loops;
}

} // namespace vesyla::transformation::stage_io
