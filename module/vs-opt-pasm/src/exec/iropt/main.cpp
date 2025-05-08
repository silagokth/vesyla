//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "pasm/Dialect.hpp"
#include "pasm/Passes.hpp"
#include "tm/Solver.hpp"
#include "tm/TimingModel.hpp"
#include "util/Common.hpp"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {

  vesyla::tm::TimingModel tm;
  tm.add_operation(vesyla::tm::Operation("op0", "T<2>(e0, e1)"));
  tm.add_operation(vesyla::tm::Operation("op1", "R<2, t3>(e0)"));
  tm.add_constraint(vesyla::tm::Constraint("linear", "op0.e0 > op1.e0"));
  tm.add_constraint(vesyla::tm::Constraint("linear", "op0.e0 > op1.e0"));
  tm.add_constraint(
      vesyla::tm::Constraint("linear", "op0.e0 > op1.e0[1]+t3-1"));
  tm.compile();
  LOG(INFO) << tm.to_mzn();
  vesyla::tm::Solver solver;
  unordered_map<string, string> result = solver.solve(tm);
  for (auto it = result.begin(); it != result.end(); ++it) {
    LOG(INFO) << it->first << " = " << it->second;
  }

  // mlir::registerAllPasses();
  // vesyla::pasm::registerPasses();

  // mlir::DialectRegistry registry;
  // registry.insert<vesyla::pasm::PasmDialect>();

  // return mlir::asMainReturnCode(
  //     mlir::MlirOptMain(argc, argv, "CIDFG optimizer driver\n", registry));
}
