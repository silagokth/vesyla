#ifndef VESYLA_TRANSFORMATION_DOUBLE_BUFFER_PASSES
#define VESYLA_TRANSFORMATION_DOUBLE_BUFFER_PASSES

include "mlir/Pass/PassBase.td"

def DoubleBufferPass : Pass<"double-buffer-pass", "::mlir::ModuleOp"> {
  let summary = "Give a staging buffer refilled every pass two slots, so the "
                "next fetch can overlap the reads of the current pass.";
  let description = [{
    Runs after StageIoPass. A staging buffer whose rows are fetched from the
    io in the body of a loop L, and read only in a loop directly inside that
    body, waits every pass: the next fetch would overwrite the row the
    current pass still reads.

    L (constant trip count n, even, no iter_args) is split into jo (n/2) and
    ji (2), L's counter becoming 2 * jo + ji, and the buffer gets a second
    slot: the fetch and the reads of a pass use slot ji. Passes of ji then
    touch different rows, so the scheduler only orders a pass after the one
    two passes earlier, and the fetch can run during the previous pass.

    Off with vs-compile --no-double-buffer; vs-compile also compiles without
    it when the extra rows do not fit.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // VESYLA_TRANSFORMATION_DOUBLE_BUFFER_PASSES
