#ifndef __VESYLA_IROPT_CIDFG_PASSES_HPP__
#define __VESYLA_IROPT_CIDFG_PASSES_HPP__

#include "cidfg/Dialect.hpp"
#include "cidfg/Ops.hpp"
#include "mlir/Pass/Pass.h"

namespace vesyla {
namespace cidfg {
#define GEN_PASS_DECL
#include "cidfg/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "cidfg/Passes.hpp.inc"
} // namespace cidfg
} // namespace vesyla

#endif // __VESYLA_IROPT_CIDFG_PASSES_HPP__