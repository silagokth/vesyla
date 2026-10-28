#ifndef VESYLA_TRANSFORMATION_STAGE_IO_PASSES
#define VESYLA_TRANSFORMATION_STAGE_IO_PASSES

include "mlir/Pass/PassBase.td"

def StageIoPass : Pass<"stage-io-pass", "::mlir::ModuleOp"> {
  let summary = "Move io rows through a staging buffer instead of accessing "
                "the io buffers word by word.";
  let description = [{
    The io resource moves whole rows -- one 16-word bulk transfer between the
    io buffer and the fabric -- and the io patterns only match those. A kernel
    lowered from C reads and writes the io buffers word by word, which no
    resource can do.

    Every scalar `affine.load` on `@input_buffer` reads from a staging buffer
    instead (`memref<16xi16>`, an RF once bound), and its row is fetched into
    that buffer with an `affine.vector_load` / `affine.vector_store` pair, at
    the outermost place where the row index no longer changes: right before the
    loop that walks the words of the row. Every scalar `affine.store` on
    `@output_buffer` writes to a staging buffer, and the whole row is written
    out right after that loop. Accesses to the same row at the same place share
    one staging buffer.

    A loop whose counter `x` picks the row as `x floordiv 16` and the word as
    `x mod 16` (a matrix row spanning several io rows, as in mvm) is first
    strip-mined into a loop over rows and a loop over the 16 words, so the row
    stays fixed inside the inner one. A loop of at most 16 iterations only has
    its `floordiv` / `mod` folded away.

    Reported as errors, for now:
      - a read whose row changes with the word index (a sliding window,
        `a[i + j]`; C frontend G17),
      - a write that covers only part of a row (C frontend G16),
      - a write or read whose row changes inside the loop over its words
        (`out[col][row]`),
      - a read whose row changes inside a reduction loop longer than a row
        (mvm, mmm with K > 16): split, the accumulation would run over two
        loops (C frontend G18).
    No word-by-word io access is left afterwards, so none can fall through to
    the RF patterns. A program that only moves whole rows -- every
    hand-written one -- passes through unchanged.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::vector::VectorDialect"];
}

#endif // VESYLA_TRANSFORMATION_STAGE_IO_PASSES
