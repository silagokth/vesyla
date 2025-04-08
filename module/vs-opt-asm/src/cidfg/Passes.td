#ifndef VESYLA_IROPT_CIDFG_PASSES
#define VESYLA_IROPT_CIDFG_PASSES

include "mlir/Pass/PassBase.td"

def CidfgConstantFolding: Pass<"cidfg-constant-folding", "::mlir::ModuleOp"> {
  let summary = "Fold constant integers";
  let description = [{
    This pass folds constant integers in the IR.
    ```
  }];

  
}

#endif // VESYLA_IROPT_CIDFG_PASSES