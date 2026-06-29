#ifndef VESYLA_CONVERSION_DRRA_TO_PASM_PASSES
#define VESYLA_CONVERSION_DRRA_TO_PASM_PASSES

include "mlir/Pass/PassBase.td"

def DrraToPasmPass : Pass<"drra-to-pasm-pass", "::mlir::ModuleOp"> {
  let summary = "Lower drra.rop into pasm.rop and pasm.instr.";
  let description = [{
    Converts each `drra.rop` into a `pasm.rop` carrying a body of
    `pasm.instr` operations. TODO: describe the mapping.
  }];
  let dependentDialects = ["::vesyla::drra::DrraDialect",
                           "::vesyla::pasm::PasmDialect"];
}

#endif // VESYLA_CONVERSION_DRRA_TO_PASM_PASSES
