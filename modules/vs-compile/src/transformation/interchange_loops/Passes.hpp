#ifndef __VESYLA_TRANSFORMATION_INTERCHANGE_LOOPS_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_INTERCHANGE_LOOPS_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace interchange_loops {
#define GEN_PASS_DECL
#include "transformation/interchange_loops/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/interchange_loops/Passes.hpp.inc"

} // namespace interchange_loops
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_INTERCHANGE_LOOPS_PASSES_HPP__
