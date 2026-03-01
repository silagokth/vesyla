#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

class AffineAnalyzerPass
    : public PassWrapper<AffineAnalyzerPass, OperationPass<mlir::ModuleOp>> {
private:
  void runOnOperation() override;

  StringRef getArgument() const final { return "affine-analyzer"; }
  StringRef getDescription() const final {
    return "Analyze affine for loops to produce a structured representation of "
           "the loop nest.";
  }
};
