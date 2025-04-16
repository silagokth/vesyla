//===- StandalonePasses.h - Standalone passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __VESYLA_IROPT_PASM_PASSES_HPP__
#define __VESYLA_IROPT_PASM_PASSES_HPP__

#include "mlir/Pass/Pass.h"
#include "pasm/Dialect.hpp"
#include "pasm/Ops.hpp"
#include <memory>

namespace vesyla {
namespace pasm {
#define GEN_PASS_DECL
#include "pasm/Passes.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "pasm/Passes.hpp.inc"
} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_IROPT_PASM_PASSES_HPP__