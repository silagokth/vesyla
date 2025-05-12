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
  tm.from_string(
      R"(
operation read_b R<5, t3>(R<3, t2>(e0))
operation read_a R<5, t1>(R<3, t0>(e0))
constraint linear read_a==read_b
constraint linear read_a.e0[0]==read_b.e0[0]
constraint linear read_a.e0[1]==read_b.e0[1]
constraint linear read_a.e0[1][2]==read_b.e0[1][2]
)");
  vesyla::tm::Solver solver;
  unordered_map<string, string> result = solver.solve(tm);

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

  vesyla::pasm::ReplaceEpochOpOptions replace_epoch_op_options = {
      .component_map = R"({"0_0_1_1":"rf", "0_0_1_0":"rf"})",
      .component_path = VESYLA_SUITE_PATH_COMPONENTS,
      .tmp_path = VESYLA_SUITE_PATH_TMP};
  ::mlir::registerPass(
      [&replace_epoch_op_options]() -> std::unique_ptr<::mlir::Pass> {
        return vesyla::pasm::createReplaceEpochOp(replace_epoch_op_options);
      });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return vesyla::pasm::createReplaceLoopOp();
  });

  mlir::DialectRegistry registry;
  registry.insert<vesyla::pasm::PasmDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CIDFG optimizer driver\n", registry));
}
