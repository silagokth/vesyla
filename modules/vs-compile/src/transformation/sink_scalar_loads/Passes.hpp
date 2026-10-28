#ifndef __VESYLA_TRANSFORMATION_SINK_SCALAR_LOADS_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_SINK_SCALAR_LOADS_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace sink_scalar_loads {
#define GEN_PASS_DECL
#include "transformation/sink_scalar_loads/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/sink_scalar_loads/Passes.hpp.inc"

} // namespace sink_scalar_loads
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_SINK_SCALAR_LOADS_PASSES_HPP__
