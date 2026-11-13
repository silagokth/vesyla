#ifndef __VESYLA_TRANSFORMATION_STAGE_IO_IO_ACCESS_HPP__
#define __VESYLA_TRANSFORMATION_STAGE_IO_IO_ACCESS_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace vesyla::transformation::stage_io {

// A word-by-word access to an io buffer: the map's first result picks the
// row, the second the word in it.
struct IoAccess {
  mlir::Operation *op;
  bool write;
  mlir::Value buffer;
  mlir::AffineMap map;
  llvm::SmallVector<mlir::Value> operands;

  mlir::AffineExpr row() const { return map.getResult(0); }
  mlir::AffineExpr word() const { return map.getResult(1); }
};

std::optional<IoAccess> as_io_access(mlir::Operation *op);

llvm::SmallVector<IoAccess> collect_io_accesses(mlir::Operation *scope);

// The loops whose counters `e` reads.
llvm::SmallVector<mlir::affine::AffineForOp> loops_of(const IoAccess &access,
                                                      mlir::AffineExpr e);

} // namespace vesyla::transformation::stage_io

#endif // __VESYLA_TRANSFORMATION_STAGE_IO_IO_ACCESS_HPP__
