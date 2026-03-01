#include "pasm/Dialect.hpp"
#include "tm/AffineAnalyzer.hpp"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char *argv[]) {
  mlir::DialectRegistry registry;
  registry.insert<vesyla::pasm::PasmDialect>();
  mlir::registerAllDialects(registry);

  llvm::errs() << "Registering AffineAnalyzerPass\n";
  mlir::PassRegistration<AffineAnalyzerPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Affine Analyzer Pass\n", registry));
}
