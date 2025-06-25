#ifndef __VESYLA_PASM_OPS_HPP__
#define __VESYLA_PASM_OPS_HPP__

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
using namespace mlir;
namespace vesyla {
namespace pasm {} // namespace pasm
} // namespace vesyla

#include "pasm/Types.hpp"

#define GET_OP_CLASSES
#include "pasm/Ops.hpp.inc"

namespace vesyla {
namespace pasm {} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_OPS_HPP__
