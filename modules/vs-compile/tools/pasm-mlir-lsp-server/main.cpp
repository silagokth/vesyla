#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "vesyla/Dialect/Drra/IR/DrraDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<vesyla::pasm::PasmDialect>();
  registry.insert<vesyla::drra::DrraDialect>();
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
