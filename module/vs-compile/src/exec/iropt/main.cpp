//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

#include "pasm/CodeGen.hpp"
#include "pasm/Dialect.hpp"
#include "pasm/Passes.hpp"
#include "util/Common.hpp"

#include "pasm/Parser.hpp"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {

  if (argc < 2) {
    llvm::errs() << "Usage: " << argv[0] << " <input.mlir>\n";
    return 1;
  }

  ifstream input_file(argv[1]);
  if (!input_file.is_open()) {
    llvm::errs() << "Error opening input file: " << argv[1] << "\n";
    return 1;
  }
  std::string input_str((std::istreambuf_iterator<char>(input_file)),
                        std::istreambuf_iterator<char>());
  input_file.close();

  vesyla::pasm::Parser parser;
  mlir::ModuleOp *module = parser.parse(input_str);
  if (module == nullptr) {
    llvm::errs() << "Error parsing input file: " << argv[1] << "\n";
    return 1;
  }

  // // Initialize MLIR context and register dialects
  // mlir::MLIRContext context;
  // mlir::DialectRegistry registry;
  // registry.insert<vesyla::pasm::PasmDialect>();
  // context.appendDialectRegistry(registry);

  // // Parse the input file into a ModuleOp
  // mlir::OwningOpRef<mlir::ModuleOp> module =
  //     mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  // if (!module) {
  //   llvm::errs() << "Error parsing input file.\n";
  //   return 1;
  // }

  // get the environment variable: VESYLA_SUITE_PATH_COMPONENTS
  std::string VESYLA_SUITE_PATH_COMPONENTS =
      std::getenv("VESYLA_SUITE_PATH_COMPONENTS")
          ? std::getenv("VESYLA_SUITE_PATH_COMPONENTS")
          : "";
  if (VESYLA_SUITE_PATH_COMPONENTS == "") {
    llvm::outs() << "Error: VESYLA_SUITE_PATH_COMPONENTS is not set.\n";
    std::exit(-1);
  }

  // get the environment variable: VESYLA_SUITE_PATH_TMP
  std::string VESYLA_SUITE_PATH_TMP = std::getenv("VESYLA_SUITE_PATH_TMP")
                                          ? std::getenv("VESYLA_SUITE_PATH_TMP")
                                          : "";
  if (VESYLA_SUITE_PATH_TMP == "") {
    llvm::outs() << "Error: VESYLA_SUITE_PATH_TMP is not set.\n";
    std::exit(-1);
  }
  // create the directory if it does not exist
  if (mkdir(VESYLA_SUITE_PATH_TMP.c_str(), 0777) == -1) {
    if (errno != EEXIST) {
      llvm::outs() << "Error: Failed to create directory: "
                   << VESYLA_SUITE_PATH_TMP << "\n";
      std::exit(-1);
    }
  }

  // Create a PassManager
  mlir::PassManager pm(&context);

  // Add passes to the pipeline
  pm.addPass(vesyla::pasm::createReplaceEpochOp(
      {.component_map =
           R"({"0_0_1_1":"rf", "0_0_1_0":"rf", "1_0_1_1":"rf", "1_0_1_0":"rf"})",
       .component_path = VESYLA_SUITE_PATH_COMPONENTS,
       .tmp_path = VESYLA_SUITE_PATH_TMP,
       .row = 2,
       .col = 2}));
  pm.addPass(vesyla::pasm::createReplaceLoopOp());
  pm.addPass(vesyla::pasm::createMergeRawOp());
  pm.addPass(vesyla::pasm::createAddHaltPass());

  // Run the pass pipeline
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Pass pipeline failed.\n";
    return 1;
  }

  // Print the transformed module to stdout
  module->print(llvm::outs());

  // Save the transformed module to a file
  std::string output_filename = "output";
  std::string output_dir = ".";
  vesyla::pasm::CodeGen cg;
  cg.generate(module.get(), output_dir, output_filename);
  llvm::outs() << "Generated code in " << output_dir << "/" << output_filename
               << "\n";

  return 0;
}
