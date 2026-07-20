#ifndef __VESYLA_DRRA_OPS_HPP__
#define __VESYLA_DRRA_OPS_HPP__

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
using namespace mlir;

#define GET_OP_CLASSES
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp.inc"

#endif // __VESYLA_DRRA_OPS_HPP__
