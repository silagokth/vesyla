//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cidfg/Dialect.hpp"
#include "cidfg/Ops.hpp"
#include "cidfg/Types.hpp"

using namespace mlir;
using namespace vesyla::cidfg;

#include "cidfg/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void CidfgDialect::initialize() {
  registerOps();
  registerTypes();
}
