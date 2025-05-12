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
    Option<"tmp_path", "tmp-path", "std::string", "\"\"", "">
  ];
}

def ReplaceLoopOp: Pass<"replace-loop-op", "::mlir::ModuleOp"> {
  let summary = "Replace the for loop structure with normal epoch regions.";
  let description = [{
    Replace the for loop structure with normal epoch regions.
  }];
}

#endif // VESYLA_IROPT_PASM_PASSES