#ifndef __VESYLA_TRANSFORMATION_STAGE_IO_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_STAGE_IO_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace stage_io {
#define GEN_PASS_DECL
#include "transformation/stage_io/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/stage_io/Passes.hpp.inc"

} // namespace stage_io
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_STAGE_IO_PASSES_HPP__
