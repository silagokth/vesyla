#ifndef VESYLA_TRANSFORMATION_COPY_STORAGES_PASSES
#define VESYLA_TRANSFORMATION_COPY_STORAGES_PASSES

include "mlir/Pass/PassBase.td"

def CopyStoragesPass : Pass<"copy-storages-pass", "::mlir::ModuleOp"> {
  let summary = "Give a storage read twice in the same cycle a copy per reader.";
  let description = [{
    Runs between SplitEpochsPass and BindResourcesPass. Two reads of one
    storage that the data flow ties to the same cycle on the same port cannot
    both be served by one register file. Every further reader gets a copy of
    the storage, filled by a clone of each fetch (io read and RF write) that
    fills the original. Copies are shared between epochs. A storage written by
    anything other than a fetch, or written twice in one cycle, is a compile
    error.
  }];
  let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

#endif // VESYLA_TRANSFORMATION_COPY_STORAGES_PASSES
