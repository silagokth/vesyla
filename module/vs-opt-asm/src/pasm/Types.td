#ifndef VESYLA_IROPT_PASM_TYPE
#define VESYLA_IROPT_PASM_TYPE

include "pasm/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

class Pasm_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Pasm_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def InstrType : Pasm_Type<"Instr", "instr"> {
  let description = [{
    The type of an instruction.
  }];
}

#endif // VESYLA_IROPT_PASM_TYPE
