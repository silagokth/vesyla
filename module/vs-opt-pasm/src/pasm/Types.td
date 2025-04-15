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

def RopType : Pasm_Type<"Rop", "rop"> {
  let description = [{
    The type of a resource operation.
  }];
}

def CopType : Pasm_Type<"Cop", "cop"> {
  let description = [{
    The type of a control operation.
  }];
}

def RawType : Pasm_Type<"Raw", "raw"> {
  let description = [{
    The type of a raw operation.
  }];
}

def AnchorType : Pasm_Type<"Anchor", "anchor"> {
  let description = [{
    The type of an anchor.
  }];
}

def OprType :  AnyTypeOf<[RopType]>;

#endif // VESYLA_IROPT_PASM_TYPE
