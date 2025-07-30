//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cdfg/Ops.hpp"
#include "cdfg/Dialect.hpp"
#include "cdfg/Types.hpp"

using namespace vesyla::cdfg;

#define GET_OP_CLASSES
#include "cdfg/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

void CdfgDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cdfg/Ops.cpp.inc"
      >();
}
