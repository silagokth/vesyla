#ifndef VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES
#define VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES

include "mlir/Pass/PassBase.td"

def DesignSpaceExplorationPass : Pass<"design-space-exploration-pass", "::mlir::ModuleOp"> {
  let summary = "Assign every selected instruction to a hardware resource.";
  let description = [{
    Runs on the module instruction selection produced, where every `drra.rop`
    says *what* it does -- its `kind`, and the `conf`/`evt` segments of its
    instruction -- but nothing says *where* it runs. This pass answers that, and
    the module it leaves behind carries a `resource` on every operation.

    The three classical decisions sit here together because they are not
    independent: which resources exist (allocation), which instance each
    operation runs on (binding), and in what order the operations sharing an
    instance run (scheduling). Binding a register file fixes the endpoints of
    every access to it, so choosing well means choosing them jointly -- which is
    why they are one pass and not three, and why a search over whole solutions
    is what this is built to grow into.

    Scheduling here does not mean cycle assignment -- ScheduleEpochPass further
    down already solves exact timing against each resource's timing model. It
    means the relative order, and it is wanted for one reason: it is what says
    which operations could share an instance. Two of them may go on the same
    resource exactly when the schedule keeps them apart, so scheduling runs
    before binding and hands it that answer.

    As of now only allocation is real: it is read from the architecture file
    passed to the executable rather than decided, since that file already fixes
    which resources exist and where they sit. Scheduling is not written yet, so
    binding has nothing telling it what is contended and falls back on putting
    every operation on the first instance of its kind.

    Everything this pass decides is annotated onto the operations, so a run is
    read back from the module it leaves behind.

    The pass ends by checking that every `drra.rop` came out bound. That check
    is the contract with everything downstream -- GenerateIcdepPass and
    DrraToPasmPass both read `resource` and cannot proceed without it -- so a
    gap is reported here, against the operation that has it, rather than
    surfacing as a confusing failure several passes later.
  }];

  let dependentDialects = ["::vesyla::drra::DrraDialect",
                           "::vesyla::pasm::PasmDialect"];
}

#endif // VESYLA_TRANSFORMATION_DESIGN_SPACE_EXPLORATION_PASSES
