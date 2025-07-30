//===- StandaloneTypes.h - Standalone dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __VESYLA_CDFG_TYPES_HPP__
#define __VESYLA_CDFG_TYPES_HPP__

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include <string>

#define GET_TYPEDEF_CLASSES
#include "cdfg/Types.hpp.inc"

#endif // __VESYLA_CDFG_TYPES_HPP__
