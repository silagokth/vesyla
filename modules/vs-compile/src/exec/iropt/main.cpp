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

#include "pasmpar/Parser.hpp"

#include "schedule/Scheduler.hpp"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {

  if (argc < 4) {
    llvm::errs() << "Usage: " << argv[0] << " <input.mlir> <arch> <isa>\n";
    return 1;
  }

  vesyla::pasmpar::Parser parser;
  std::string filename(argv[1]);
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<vesyla::pasm::PasmDialect>();
  context.appendDialectRegistry(registry);
  context.getOrLoadDialect<vesyla::pasm::PasmDialect>();
  mlir::ModuleOp module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  parser.parse(filename, &module);

  vesyla::schedule::Scheduler scheduler(argv[2], argv[3]);
  scheduler.run(module);

  return 0;
}
