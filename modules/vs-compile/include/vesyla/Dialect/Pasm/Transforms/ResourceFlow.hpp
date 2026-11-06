#ifndef __VESYLA_PASM_RESOURCE_FLOW_HPP__
#define __VESYLA_PASM_RESOURCE_FLOW_HPP__

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

// Where a drra.rop's values come from and go to in the bound architecture:
// the resource of each result and operand, and the resources a value reaches
// through loop iter_args. GenerateIcdepPass builds the icdeps from these;
// SplitEpochsPass counts the interconnect options they take.
namespace vesyla::pasm {

// The resource of result `result_index` of `op`. Ops with a single endpoint
// carry a plain ResourceAttr; ops with multiple inputs and an output carry an
// array laid out as [outputs..., inputs...].
ResourceAttr get_result_resource(mlir::Operation *op, unsigned result_index);

// The resource of operand `operand_index` of `op`, at index
// num_results + operand_index of the array layout.
ResourceAttr get_operand_resource(mlir::Operation *op, unsigned operand_index);

// The resources of every resourced consumer `value` reaches, looking through
// affine.for iter_args and affine.yield. `visited` guards the iter_arg <->
// yield cycle.
void collect_consumer_resources(mlir::Value value,
                                llvm::SmallVectorImpl<mlir::Attribute> &dst,
                                llvm::SmallPtrSetImpl<void *> &visited);

// The data kind a value travels as: "bulk" for a vector, "word" for a scalar.
llvm::StringRef kind_of(mlir::Value value);

} // namespace vesyla::pasm

#endif // __VESYLA_PASM_RESOURCE_FLOW_HPP__
