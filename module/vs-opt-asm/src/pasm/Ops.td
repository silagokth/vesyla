#ifndef VESYLA_IROPT_PASM_OPS
#define VESYLA_IROPT_PASM_OPS

include "pasm/Dialect.td"
include "pasm/Types.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"


class Pasm_Op<string mnemonic, list<Trait> traits = []> :
    Op<Pasm_Dialect, mnemonic, traits>;

def InstrOp : Pasm_Op<"Instr"> {
  let arguments = (ins StrAttr:$id, StrAttr:$type, I32Attr:$row, I32Attr:$col, I32Attr:$slot, I32Attr:$port);
  let results = (outs InstrType: $result);
}

#endif // VESYLA_IROPT_PASM_OPS
