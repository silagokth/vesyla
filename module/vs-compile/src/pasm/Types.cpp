//===- StandaloneTypes.cpp - Standalone dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pasm/Types.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "pasm/Dialect.hpp"
#include "llvm/ADT/TypeSwitch.h"

using namespace vesyla::pasm;

#define GET_TYPEDEF_CLASSES
#include "pasm/Types.cpp.inc"

void PasmDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "pasm/Types.cpp.inc"
      >();
}
