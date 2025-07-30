//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cdfg/Dialect.hpp"
#include "cdfg/Ops.hpp"
#include "cdfg/Types.hpp"

using namespace mlir;
using namespace vesyla::cdfg;

#include "cdfg/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void CdfgDialect::initialize() {
  registerOps();
  registerTypes();
}
