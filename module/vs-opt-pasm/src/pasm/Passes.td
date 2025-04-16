#ifndef VESYLA_IROPT_PASM_PASSES
#define VESYLA_IROPT_PASM_PASSES

include "mlir/Pass/PassBase.td"

def ReplaceLoopOp: Pass<"replace-loop-op", "::mlir::ModuleOp"> {
  let summary = "Replace the for loop structure with normal epoch regions.";
  let description = [{
    Replace the for loop structure with normal epoch regions.
  }];
}

#endif // VESYLA_IROPT_PASM_PASSES