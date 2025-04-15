//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "asm/Dialect.hpp"
#include "asm/Ops.hpp"

using namespace mlir;
using namespace vesyla::asmd;

#include "asm/Dialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "asm/Ops.cpp.inc"
#include "asm/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void ASMDialect::initialize() {
  registerOps();
  registerTypes();
}