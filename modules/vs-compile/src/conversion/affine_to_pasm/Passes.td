#ifndef VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES
#define VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES

include "mlir/Pass/PassBase.td"

def AffineToInstrPass : Pass<"affine-to-instr-pass", "::mlir::ModuleOp"> {
  let summary = "Lower innermost affine.for loops over pasm.rop into rep instrs.";
  let description = [{
    Matches each innermost `affine.for` whose body contains exactly one
    `pasm.rop` (and no nested `affine.for`). Appends a
    `pasm.instr<type="rep", param={iter, step, delay}>` derived from the
    loop's constant bounds at the end of the rop body, hoists the rop out of
    the loop, and erases the now-empty `affine.for`. Greedy iteration lowers
    nested loops innermost-out, one level per match. After all patterns are
    applied, any leftover `map` attribute on a rop is removed.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::vesyla::pasm::PasmDialect"];
}

#endif // VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES
