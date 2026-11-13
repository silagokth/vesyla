#ifndef VESYLA_TRANSFORMATION_FISSION_LOOPS_PASSES
#define VESYLA_TRANSFORMATION_FISSION_LOOPS_PASSES

include "mlir/Pass/PassBase.td"

def FissionLoopsPass : Pass<"fission-loops-pass", "::mlir::ModuleOp"> {
  let summary = "Split a loop nest that needs more DPUs than the fabric has "
                "into one nest per group of DPU operations.";
  let description = [{
    Runs between instruction selection and BindResourcesPass, which gives
    every DPU rop with a `conf` in one outermost loop nest its own DPU; a nest
    with more of them than the architecture has DPUs is split here.

    The body of the nest (of its innermost loop, for a perfect nest) is cut
    into parts: store trees are packed into a part in program order while they
    fit, and a tree that alone needs too many DPUs is cut at its operations.
    Each part becomes a copy of the loop nest, one after the other in the same
    epoch, so binding can share DPUs between them; SplitEpochsPass cuts the
    epoch later only where it has to. A value computed in one part and used in
    a later one goes through a new buffer, written and read with the RF word
    patterns of the component library. Reads, fetches and index ops are copied
    into every part that needs them. Fetches before the nest move before the
    part of their first reader.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // VESYLA_TRANSFORMATION_FISSION_LOOPS_PASSES
