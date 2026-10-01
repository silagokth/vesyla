#ifndef __VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES_HPP__

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include <memory>
#include <string>

namespace vesyla {
namespace transformation {
namespace dse {
#define GEN_PASS_DECL
#include "transformation/design_space_exploration/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/design_space_exploration/Passes.hpp.inc"

} // namespace dse
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES_HPP__
