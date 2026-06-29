#ifndef __VESYLA_CONVERSION_DRRA_TO_PASM_PASSES_HPP__
#define __VESYLA_CONVERSION_DRRA_TO_PASM_PASSES_HPP__

#include "mlir/Pass/Pass.h"
#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include <memory>

namespace vesyla {
namespace conversion {
namespace drra_to_pasm {
#define GEN_PASS_DECL
#include "conversion/drra_to_pasm/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "conversion/drra_to_pasm/Passes.hpp.inc"

} // namespace drra_to_pasm
} // namespace conversion
} // namespace vesyla

#endif // __VESYLA_CONVERSION_DRRA_TO_PASM_PASSES_HPP__
