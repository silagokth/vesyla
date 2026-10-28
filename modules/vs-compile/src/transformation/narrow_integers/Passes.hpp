#ifndef __VESYLA_TRANSFORMATION_NARROW_INTEGERS_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_NARROW_INTEGERS_PASSES_HPP__

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace vesyla {
namespace transformation {
namespace narrow_integers {
#define GEN_PASS_DECL
#include "transformation/narrow_integers/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/narrow_integers/Passes.hpp.inc"

} // namespace narrow_integers
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_NARROW_INTEGERS_PASSES_HPP__
