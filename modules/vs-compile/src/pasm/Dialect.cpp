//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pasm/Dialect.hpp"
#include "pasm/Ops.hpp"
#include "pasm/Types.hpp"

using namespace mlir;
using namespace vesyla::pasm;

#include "pasm/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void PasmDialect::initialize() {
  registerOps();
  registerTypes();
}
