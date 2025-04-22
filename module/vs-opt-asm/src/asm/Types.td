#ifndef ASM_TYPE
#define ASM_TYPE

include "asm/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

class ASM_Type<string name, string typeMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<ASM_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def InstrType : ASM_Type<"instr", "instr"> {
  let description = "Instruction type";
}
#endif // ASM_TYPE
