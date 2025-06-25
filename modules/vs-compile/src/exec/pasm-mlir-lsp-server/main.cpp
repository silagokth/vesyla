#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "pasm/Dialect.hpp"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Add the CIR dialect to the registry
  registry.insert<cir::CIRDialect>();

  // Add the Pasm dialect to the registry
  registry.insert<vesyla::pasm::PasmDialect>();

  // Register the dialects with the MLIR LSP server
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
