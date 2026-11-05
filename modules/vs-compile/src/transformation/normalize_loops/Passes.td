#ifndef VESYLA_TRANSFORMATION_NORMALIZE_LOOPS_PASSES
#define VESYLA_TRANSFORMATION_NORMALIZE_LOOPS_PASSES

include "mlir/Pass/PassBase.td"

def NormalizeLoopsPass : Pass<"normalize-loops-pass", "::mlir::ModuleOp"> {
  let summary = "Rewrite affine.for loops to start at 0 with step 1.";
  let description = [{
    A rep counts from 0 in steps of 1, so AffineToRepPass takes only such
    loops. `for i = 1 to 29` becomes `for i = 0 to 28`, with `i + 1` folded
    into the maps of the affine accesses that used `i`. Loops already in
    that form are left alone.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect"];
}

#endif // VESYLA_TRANSFORMATION_NORMALIZE_LOOPS_PASSES
