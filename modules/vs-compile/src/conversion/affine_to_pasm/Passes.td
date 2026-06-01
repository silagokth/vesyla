#ifndef VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES
#define VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES

include "mlir/Pass/PassBase.td"

def AffineToInstrPass : Pass<"affine-to-instr-pass", "::mlir::ModuleOp"> {
  let summary = "Lower innermost affine.for loops over pasm.rop into rep instrs.";
  let description = [{
    Matches each innermost `affine.for` whose body contains only `pasm.rop`s
    (and no nested `affine.for`). For every rop it appends a
    `pasm.instr<type="rep", param={iter, step, delay}>` derived from the
    loop's constant bounds at the end of the rop body and shrinks the rop's
    `map` by the consumed innermost dim. It then hoists all rops out of the
    loop (preserving order) and erases the now-empty `affine.for`. Greedy
    iteration lowers nested loops innermost-out, one level per sweep, so a loop
    only becomes innermost once its nested loops have been erased. After all
    patterns are applied, any leftover `map` attribute on a rop is removed.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::vesyla::pasm::PasmDialect"];
}

#endif // VESYLA_CONVERSION_AFFINE_TO_PASM_PASSES
