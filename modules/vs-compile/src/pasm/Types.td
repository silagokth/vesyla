#ifndef VESYLA_PASM_TYPE
#define VESYLA_PASM_TYPE

include "pasm/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

class Pasm_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Pasm_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def DummyType : Pasm_Type<"Dummy", "dummy"> {
  let description = [{
    A dummy type.
  }];
}


#endif // VESYLA_PASM_TYPE
