#ifndef __VESYLA_CONVERSION_FUNC_TO_PASM_PASSES_HPP__
#define __VESYLA_CONVERSION_FUNC_TO_PASM_PASSES_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include <memory>

namespace vesyla {
namespace conversion {
namespace func_to_pasm {
#define GEN_PASS_DECL
#include "conversion/func_to_pasm/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "conversion/func_to_pasm/Passes.hpp.inc"

} // namespace func_to_pasm
} // namespace conversion
} // namespace vesyla

#endif // __VESYLA_CONVERSION_FUNC_TO_PASM_PASSES_HPP__
