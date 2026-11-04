#ifndef VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES
#define VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES

include "mlir/Pass/PassBase.td"

def SplitEpochsPass : Pass<"split-epochs-pass", "::mlir::ModuleOp"> {
  let summary = "Cut an epoch where a DPU would need more configurations than "
                "it has config registers.";
  let description = [{
    Runs after BindResourcesPass. Every distinct DPU configuration (conf
    without its option) a DPU runs in an epoch takes one of its config
    registers. Walking the top-level operations of an epoch in order, the
    epoch is cut before the first one that would take a fifth register on
    some DPU, at the latest point before it that no SSA value crosses. The
    operations from there on go to a new epoch, which is checked the same way.
    The binding stays valid: program order is unchanged and register files
    keep their contents across epochs.
  }];
}

#endif // VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES
