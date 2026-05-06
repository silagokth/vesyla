#include "conversion/MemrefToPasm/MemrefToPasm.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

// Dialects used in the input file
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "pasm/Dialect.hpp"

// Custom conversion passes
#include "conversion/ArithToPasm/ArithToPasm.hpp"
#include "conversion/FuncToPasm/FuncToPasm.hpp"

// Custom transformation passes
#include "transformation/AddSWBPass.hpp"

// utility functions
#include "plog/Log.h"
#include "util/MiniArgs.hpp"

#include <string>

int main(int argc, char **argv) {

  vesyla::util::MiniArgs args;

  args.parse(argc, argv);
  if (args.flag("h") || args.flag("help")) {
    LOG_INFO << "Usage: pasm-opt --mlir FILE --output FILE";
    return 0;
  }

  std::string mlir_file = args.get("mlir", args.get("m"));
  std::string output_dir = args.get("output", args.get("o", "."));

  LOG_INFO << "mlir-file: " << mlir_file;

  // llvm::errs().changeColor(llvm::raw_ostream::RED, /*bold=*/true);

  if (mlir_file.empty()) {
    llvm::errs() << "[ERROR]: --mlir <file> is required\n";
    return -1;
  }

  // register dialects
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::scf::SCFDialect, vesyla::pasm::PasmDialect,
                  mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);

  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_file, &context);

  if (!module) {
    llvm::errs() << "[ERROR]: failed to parse " << mlir_file << "\n";
    return 1;
  }

  module->dump();

  // AddSWBPass transformation pass (runs first)
  {
    mlir::PassManager pm(module->getContext());
    pm.addPass(vesyla::transformation::createAddSWBPass());
    if (mlir::failed(pm.run(module.get()))) {
      LOG_FATAL << "[ERROR]: AddSWBPass pass failed.\n";
      std::exit(EXIT_FAILURE);
    }
  }
  module->dump();
  /* ArithToPasm conversion pass
  {
    mlir::PassManager pm(module->getContext());
    pm.addPass(vesyla::conversion::createArithToPasm());
    if (mlir::failed(pm.run(module.get()))) {
      LOG_FATAL << "[ERROR]: ArithToPasm pass failed.\n";
      std::exit(EXIT_FAILURE);
    }
    module->dump();
  }

  // MemrefToPasm conversion pass
  {
    mlir::PassManager pm(module->getContext());
    pm.addPass(vesyla::conversion::createMemrefToPasm());
    if (mlir::failed(pm.run(module.get()))) {
      LOG_FATAL << "Error: MemrefToPasm pass failed.\n";
      std::exit(EXIT_FAILURE);
    }
    module->dump();
  }*/

  // FuncToPasm conversion pass
  {
    mlir::PassManager pm(module->getContext());
    pm.addPass(vesyla::conversion::createFuncToPasm());
    if (mlir::failed(pm.run(module.get()))) {
      LOG_FATAL << "Error: FuncToPasm pass failed.\n";
      std::exit(EXIT_FAILURE);
    }
    module->dump();
  }

  return 0;
}
