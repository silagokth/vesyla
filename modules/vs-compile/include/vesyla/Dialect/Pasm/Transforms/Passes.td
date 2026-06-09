#ifndef VESYLA_PASM_PASSES
#define VESYLA_PASM_PASSES

include "mlir/Pass/PassBase.td"

def ScheduleEpochPass: Pass<"schedule-epoch-pass", "::mlir::ModuleOp"> {
  let summary = "Schedule the epoch structure with normal epoch regions.";
  let description = [{
    Schedule the epoch structure with normal epoch regions.
  }];
  let options = [
    Option<"component_path", "component-path", "std::string", "\"\"", "">,
    Option<"tmp_path", "tmp-path", "std::string", "\"\"", "">,
    Option<"allow_unsafe", "allow-unsafe", "bool", "false",
      "Allow non-consecutive scheduling fallback if true">,
    Option<"keep_debug_files", "keep-debug-files", "bool", "false",
      "Keep all intermediate files (no cleanup) if true">,
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

def GenerateIcdepPass: Pass<"generate-icdep-pass", "::mlir::ModuleOp"> {
  let summary = "Generate pasm.icdep ops.";
  let description = [{
    TODO: describe behavior once implemented.
  }];
}

def InterconnectPass: Pass<"interconnect-pass", "::mlir::ModuleOp"> {
  let summary = "Lower pasm.icdep edges to pasm.rop and pasm.cstr ops.";
  let description = [{
    Walks every `pasm.icdep` op and rewrites it into the resource
    operations (`pasm.rop`) and delay constraints (`pasm.cstr`) that the
    interconnect requires to realize the dependency.
  }];
}

def FlattenCellsPass: Pass<"flatten-cells-pass", "::mlir::ModuleOp"> {
  let summary = "Flatten cell contents back into the epoch body.";
  let description = [{
    Walks every `pasm.epoch` op, iterates over all contained `pasm.cell`
    ops, moves their body operations back into the epoch, and erases the
    now-empty cells.
  }];
}

def ExtractCellsPass: Pass<"extract-cells-pass", "::mlir::ModuleOp"> {
  let summary = "Extract cells from each epoch.";
  let description = [{
    Walks every `pasm.epoch` op and extracts its cells.
    TODO: describe behavior once implemented.
  }];
}
#endif // VESYLA_PASM_PASSES
