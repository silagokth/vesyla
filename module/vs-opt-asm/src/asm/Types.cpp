//===- StandaloneTypes.cpp - Standalone dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "asm/Types.hpp"
#include "asm/Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace vesyla::asmd;

#define GET_TYPEDEF_CLASSES
#include "asm/Types.cpp.inc"

void ASMDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "asm/Types.cpp.inc"
      >();
}
