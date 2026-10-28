#ifndef VESYLA_TRANSFORMATION_NARROW_INTEGERS_PASSES
#define VESYLA_TRANSFORMATION_NARROW_INTEGERS_PASSES

include "mlir/Pass/PassBase.td"

def NarrowIntegersPass : Pass<"narrow-integers-pass", "::mlir::ModuleOp"> {
  let summary = "Undo C integer promotion: compute truncated arithmetic in the "
                "narrow type.";
  let description = [{
    C computes int16 arithmetic in `int`, so a kernel lowered from C wraps
    every operation in `arith.extsi` / `arith.trunci`. No resource takes a
    32-bit operand, and the patterns in the component library are written over
    the 16-bit word.

    The low bits of `a + b`, `a - b` and `a * b` depend only on the low bits of
    `a` and `b`, so `trunci(op(extsi a, extsi b))` is `op(a, b)` in the narrow
    type, wrapping on overflow just as the C cast back does. Starting from every
    `arith.trunci`, the tree under it is rebuilt in the narrow type: an
    `arith.addi` / `arith.subi` / `arith.muli` becomes the narrow op (keeping
    its `id`), an `arith.extsi` from the narrow type becomes its source, and an
    `arith.constant` is truncated. The wide ops left without uses are erased.

    Anything else under a truncation (a division, a shift, a comparison), or a
    wide value that is also used outside one, is not narrowed, and the
    `arith.extsi` / `arith.trunci` it leaves behind is reported as an error.
    A program without extensions or truncations -- every hand-written one --
    passes through unchanged.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

#endif // VESYLA_TRANSFORMATION_NARROW_INTEGERS_PASSES
