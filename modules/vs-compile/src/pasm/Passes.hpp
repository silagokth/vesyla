#ifndef __VESYLA_PASM_PASSES_HPP__
#define __VESYLA_PASM_PASSES_HPP__

#include "mlir/Pass/Pass.h"
#include "pasm/Dialect.hpp"
#include "pasm/Ops.hpp"
#include <memory>

namespace vesyla {
namespace pasm {
#define GEN_PASS_DECL
#include "pasm/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "pasm/Passes.hpp.inc"

std::string gen_random_string(size_t length);

} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_PASSES_HPP__