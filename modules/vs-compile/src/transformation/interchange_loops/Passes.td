#ifndef VESYLA_TRANSFORMATION_INTERCHANGE_LOOPS_PASSES
#define VESYLA_TRANSFORMATION_INTERCHANGE_LOOPS_PASSES

include "mlir/Pass/PassBase.td"

def InterchangeLoopsPass : Pass<"interchange-loops-pass", "::mlir::ModuleOp"> {
  let summary = "Reorder a loop nest so every io write's row stays fixed "
                "while its words are written.";
  let description = [{
    Runs before StageIoPass. A word-by-word io write whose row changes inside
    the loop over its words (transpose's `out[col][row]` with `row` outer)
    cannot be staged a row at a time. In a perfect nest, the loops its word
    reads are moved inside the loops its row reads, if the permutation keeps
    every dependence and leaves every other io write in the nest stageable.
    Reads are not considered: StageIoPass fetches a read's row again where it
    changes. Otherwise the nest is left for StageIoPass to report.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect"];
}

#endif // VESYLA_TRANSFORMATION_INTERCHANGE_LOOPS_PASSES
