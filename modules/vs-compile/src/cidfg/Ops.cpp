//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cidfg/Ops.hpp"
#include "cidfg/Dialect.hpp"
#include "cidfg/Types.hpp"

using namespace vesyla::cidfg;

#define GET_OP_CLASSES
#include "cidfg/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

void CidfgDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cidfg/Ops.cpp.inc"
      >();
}