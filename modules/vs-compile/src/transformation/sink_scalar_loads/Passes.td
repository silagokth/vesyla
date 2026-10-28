#ifndef VESYLA_TRANSFORMATION_SINK_SCALAR_LOADS_PASSES
#define VESYLA_TRANSFORMATION_SINK_SCALAR_LOADS_PASSES

include "mlir/Pass/PassBase.td"

def SinkScalarLoadsPass : Pass<"sink-scalar-loads-pass", "::mlir::ModuleOp"> {
  let summary = "Re-read a scalar loaded outside a loop inside every loop "
                "that uses it.";
  let description = [{
    A kernel lowered from C loads a scalar parameter once and uses it on every
    iteration (axpy's `alpha`, rot's `c` and `s`). An operation inside a loop
    runs once per iteration, so its operands must come once per iteration
    too: a value loaded once outside the loop gives one element where the loop
    consumes sixteen, and constraint generation rejects the pair.

    An `affine.load` whose result is used inside an `affine.for` that does not
    hold the load is copied into each block of such uses, right before the
    first use there; the uses in that block read the copy. Its subscript stays
    constant, which instruction selection reads as the zero-stride map
    `(d0) -> (0)`: the same word on every iteration, as a hand-written program
    writes it. The original load is erased once nothing uses it.

    Two cases are compile errors: the loaded buffer is written inside the loop
    the load would move into (the copy would read a different value), or it is
    an io buffer (the io moves whole rows only). Any other integer value
    defined outside a loop and used inside it (a product computed before the
    loop) is a compile error too, except an `arith.constant`. A hand-written
    program loads inside its loops already and passes through unchanged.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // VESYLA_TRANSFORMATION_SINK_SCALAR_LOADS_PASSES
