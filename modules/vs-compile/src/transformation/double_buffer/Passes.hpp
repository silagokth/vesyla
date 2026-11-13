#ifndef __VESYLA_TRANSFORMATION_DOUBLE_BUFFER_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_DOUBLE_BUFFER_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace double_buffer {
#define GEN_PASS_DECL
#include "transformation/double_buffer/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/double_buffer/Passes.hpp.inc"

} // namespace double_buffer
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_DOUBLE_BUFFER_PASSES_HPP__
