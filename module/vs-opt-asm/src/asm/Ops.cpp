//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "asm/Ops.hpp"
#include "asm/Dialect.hpp"
#include "asm/Types.hpp"

using namespace vesyla::asmd;

#define GET_OP_CLASSES
#include "asm/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

void ASMDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "asm/Ops.cpp.inc"
      >();
}