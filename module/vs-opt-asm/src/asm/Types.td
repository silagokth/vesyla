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

// def InstructionType : ASM_Type<"Instruction", "Instruction"> {
//   let description = [{
//     An instruction type.
//   }];
//   let parameters = (ins "unsigned": $width, StringRefParameter<"the custom value">: $bind);
//   let assemblyFormat = "`<` $width `,`  $bind `>`";
// }

#endif // ASM_TYPE
