#ifndef __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES_HPP__
#define __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include <memory>
#include <string>

namespace vesyla {
namespace conversion {
namespace select_instructions {
#define GEN_PASS_DECL
#include "conversion/select_instructions/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "conversion/select_instructions/Passes.hpp.inc"

} // namespace select_instructions
} // namespace conversion
} // namespace vesyla

#endif // __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES_HPP__
