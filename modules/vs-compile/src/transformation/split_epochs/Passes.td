#ifndef VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES
#define VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES

include "mlir/Pass/PassBase.td"

def SplitEpochsPass : Pass<"split-epochs-pass", "::mlir::ModuleOp"> {
  let summary = "Split a loop nest that needs more DPUs than the fabric has "
                "into one epoch per group of DPU operations.";
  let description = [{
    Runs between instruction selection and design space exploration. DSE gives
    every DPU rop with a `conf` in one outermost loop nest its own DPU; a nest
    with more of them than the architecture has DPUs is split here.

    The body of the nest (of its innermost loop, for a perfect nest) is cut
    into parts: store trees are packed into a part in program order while they
    fit, and a tree that alone needs too many DPUs is cut at its operations.
    Each part becomes its own epoch with a copy of the loop nest. A value
    computed in one part and used in a later one goes through a new buffer,
    written and read with the RF word patterns of the component library.
    Reads, fetches and index ops are copied into every part that needs them.
    Fetches before the nest move to the epoch of their first reader, ops after
    it to the last epoch.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // VESYLA_TRANSFORMATION_SPLIT_EPOCHS_PASSES
