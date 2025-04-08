//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "cidfg/Dialect.hpp"

int main(int argc, char **argv) {

  mlir::DialectRegistry registry;
  registry.insert<vesyla::cidfg::CidfgDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CIDFG optimizer driver\n", registry));
}
