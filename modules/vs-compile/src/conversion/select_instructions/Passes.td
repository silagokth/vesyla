#ifndef VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES
#define VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES

include "mlir/Pass/PassBase.td"

def SelectInstructionsPass : Pass<"select-instructions-pass", "::mlir::ModuleOp"> {
  let summary = "Select drra instructions from affine + arith ops.";
  let description = [{
    Pattern-based instruction selection: rewrites upstream `affine`/`arith`
    fragments into `drra.rop` operations. The patterns are described in
    `patterns.pdll` (compiled ahead-of-time into this pass with `mlir-pdll
    -x=cpp`); native match/rewrite helpers live in `NativeHelpers.cpp`.

    Phase 1 scope:
      - affine.load / affine.vector_load        -> drra.rop kind="rf" instr="evt"
      - affine.store / affine.vector_store      -> drra.rop kind="rf" instr="evt"
      - arith.muli feeding arith.addi (a loop   -> one drra.rop kind="dpu"
        iter_args reduction accumulate)             instr="conf" (a MAC)

    `id` and `resource` are copied from the source op's annotations; `kind`,
    `instr` and `param` come from the matched pattern; `map` is lifted from the
    affine access. Loops (`affine.for`) and `memref` are left untouched; resource
    binding/placement is handled by a separate later step.

    After the greedy rewrite, dead `affine.apply` ops (whose maps were lifted
    onto the rops) are erased and a completeness check fails the pass if any
    arith/affine-access op was left unselected.
  }];
  let dependentDialects = ["::vesyla::drra::DrraDialect",
                           "::vesyla::pasm::PasmDialect",
                           "::mlir::affine::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::pdl::PDLDialect",
                           "::mlir::pdl_interp::PDLInterpDialect"];
}

#endif // VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PASSES
