#ifndef __VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES_HPP__
#define __VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES_HPP__

#include "mlir/Pass/Pass.h"
#include "pasm/Dialect.hpp"
#include "pasm/Ops.hpp"
#include <memory>

namespace vesyla {
namespace conversion {
namespace affine_to_pasm {
#define GEN_PASS_DECL
#include "conversion/affine_to_pasm/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "conversion/affine_to_pasm/Passes.hpp.inc"

} // namespace affine_to_pasm
} // namespace conversion
} // namespace vesyla

#endif // __VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES_HPP__
