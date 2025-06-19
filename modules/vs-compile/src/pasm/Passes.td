#ifndef VESYLA_IROPT_PASM_PASSES
#define VESYLA_IROPT_PASM_PASSES

include "mlir/Pass/PassBase.td"

def ScheduleEpochPass: Pass<"schedule-epoch-pass", "::mlir::ModuleOp"> {
  let summary = "Schedule the epoch structure with normal epoch regions.";
  let description = [{
    Schedule the epoch structure with normal epoch regions.
  }];
  let options = [
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

def AddSlotPortPass: Pass<"add-slot-port-pass", "::mlir::ModuleOp"> {
  let summary = "Add slot and port operations.";
  let description = [{
    Add slot and port operations to the module.
  }];
}
def AddDefaultValuePass: Pass<"add-slot-port-pass", "::mlir::ModuleOp"> {
  let summary = "Add slot and port operations.";
  let description = [{
    Add slot and port operations to the module.
  }];
}
#endif // VESYLA_IROPT_PASM_PASSES