#ifndef __VESYLA_TRANSFORMATION_BIND_RESOURCES_PASSES_HPP__
#define __VESYLA_TRANSFORMATION_BIND_RESOURCES_PASSES_HPP__

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include <memory>
#include <string>

namespace vesyla {
namespace transformation {
namespace bind_resources {
#define GEN_PASS_DECL
#include "transformation/bind_resources/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "transformation/bind_resources/Passes.hpp.inc"

} // namespace bind_resources
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_BIND_RESOURCES_PASSES_HPP__
