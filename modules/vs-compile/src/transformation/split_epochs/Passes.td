#ifndef VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES
#define VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES

include "mlir/Pass/PassBase.td"

def SplitEpochsPass : Pass<"split-epochs-pass", "::mlir::ModuleOp"> {
  let summary = "Cut an epoch where a resource would need more configurations "
                "than it has options.";
  let description = [{
    Runs after BindResourcesPass. Every distinct configuration (conf without
    its option) a resource instance runs in an epoch takes one of its
    NUM_OPTIONS (arch.json; 4 when not given, 0 for no limit). Walking the
    top-level operations of an epoch in order, the epoch is cut before the
    first one that would exceed that on some resource, at the latest point
    before it that no SSA value crosses. It is cut the same way before an
    operation whose rops on some port could no longer form one AGU program
    with the rops before it (a loop shared by some of them only; the rule
    CoalesceRopsPass merges by, vesyla/Support/AguMerge.hpp). The
    operations from there on go to a new epoch, which is checked the same way.
    The binding stays valid: program order is unchanged and register files
    keep their contents across epochs.
  }];
}

#endif // VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES
