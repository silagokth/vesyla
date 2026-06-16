#ifndef VESYLA_DRRA_OPS
#define VESYLA_DRRA_OPS

include "vesyla/Dialect/Drra/IR/DrraDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

class Drra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Drra_Dialect, mnemonic, traits>;

def RopOp : Drra_Op<"rop"> {
  let summary = "Generic resource operation with variadic inputs and outputs";
  let description = [{
    A rop carries zero or more input values and produces zero or more
    output values. It does not fix any named attribute; all attributes are
    variable and stored in the op's attribute dictionary.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
}

#endif // VESYLA_DRRA_OPS
