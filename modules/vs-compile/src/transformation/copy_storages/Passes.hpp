#ifndef __VESYLA_TRANSFORMATION_COPY_STORAGES_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_COPY_STORAGES_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace copy_storages {
#define GEN_PASS_DECL
#include "transformation/copy_storages/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/copy_storages/Passes.hpp.inc"

} // namespace copy_storages
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_COPY_STORAGES_PASSES_HPP__
