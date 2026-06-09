#ifndef __VESYLA_PASM_PASSES_HPP__
#define __VESYLA_PASM_PASSES_HPP__

#include "mlir/Pass/Pass.h"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include <memory>

namespace vesyla {
namespace pasm {
#define GEN_PASS_DECL
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

std::string gen_random_string(size_t length);

} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_PASSES_HPP__