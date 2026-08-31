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
    output values.

    `conf` and `evt` hold the instruction segments -- keyed by segment name, as
    the resource's isa.json declares them -- that the later lowering needs to
    build a pasm.instr. Either may be absent, meaning that resource needs no
    instruction of that kind for this operation: a DPU mode is a conf with no
    evt, an RF access is an evt with no conf, and an operation that has to be
    configured and then triggered carries both. Note that the two do not stay
    together downstream: DrraToPasmPass hoists a conf out of the enclosing loop
    to before the earliest producer of its operands, while an evt stays where
    the rop was.

    Every other attribute is free-form and lives in the op's attribute
    dictionary.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       OptionalAttr<DictionaryAttr>:$conf,
                       OptionalAttr<DictionaryAttr>:$evt);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $outputs)
  }];
}

#endif // VESYLA_DRRA_OPS
