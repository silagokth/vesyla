#ifndef __VESYLA_TRANSFORMATION_NORMALIZE_LOOPS_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_NORMALIZE_LOOPS_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace normalize_loops {
#define GEN_PASS_DECL
#include "transformation/normalize_loops/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/normalize_loops/Passes.hpp.inc"

} // namespace normalize_loops
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_NORMALIZE_LOOPS_PASSES_HPP__
