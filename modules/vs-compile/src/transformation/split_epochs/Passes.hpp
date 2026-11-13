#ifndef __VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace split_epochs {
#define GEN_PASS_DECL
#include "transformation/split_epochs/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/split_epochs/Passes.hpp.inc"

} // namespace split_epochs
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES_HPP__
