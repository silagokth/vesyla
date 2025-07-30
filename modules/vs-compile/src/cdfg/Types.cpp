//===- StandaloneTypes.cpp - Standalone dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cdfg/Types.hpp"
#include "cdfg/Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace vesyla::cdfg;

#define GET_TYPEDEF_CLASSES
#include "cdfg/Types.cpp.inc"

void CdfgDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cdfg/Types.cpp.inc"
      >();
}
