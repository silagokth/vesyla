//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pasm/Ops.hpp"
#include "pasm/Dialect.hpp"
#include "pasm/Types.hpp"

using namespace vesyla::pasm;

#define GET_OP_CLASSES
#include "pasm/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

void PasmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "pasm/Ops.cpp.inc"
      >();
}