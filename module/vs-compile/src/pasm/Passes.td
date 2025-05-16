#ifndef VESYLA_IROPT_PASM_PASSES
#define VESYLA_IROPT_PASM_PASSES

include "mlir/Pass/PassBase.td"

def ReplaceEpochOp: Pass<"replace-epoch-op", "::mlir::ModuleOp"> {
  let summary = "Replace the epoch structure with normal epoch regions.";
  let description = [{
    Replace the epoch structure with normal epoch regions.
  }];
  let options = [
    Option<"component_map", "component-map", "std::string", "\"\"", "">,
    Option<"component_path", "component-path", "std::string", "\"\"", "">,
    Option<"tmp_path", "tmp-path", "std::string", "\"\"", "">,
    Option<"row", "row", "int", "0", "">,
    Option<"col", "col", "int", "0", "">
  ];
}

def ReplaceLoopOp: Pass<"replace-loop-op", "::mlir::ModuleOp"> {
  let summary = "Replace the for loop structure with normal epoch regions.";
  let description = [{
    Replace the for loop structure with normal epoch regions.
  }];
}

def MergeRawOp: Pass<"merge-raw-op", "::mlir::ModuleOp"> {
  let summary = "Merge raw operations into a single operation.";
  let description = [{
    Merge multiple raw operations into a single operation for optimization.
  }];
}

def AddHaltPass: Pass<"add-halt-pass", "::mlir::ModuleOp"> {
  let summary = "Add a halt operation.";
  let description = [{
    Add a halt operation to terminate execution.
  }];
}

#endif // VESYLA_IROPT_PASM_PASSES