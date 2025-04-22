//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "asm/Dialect.hpp"
#include "asm/Ops.hpp"
#include "asm/Types.hpp"

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

Type ASMDialect::parseType(DialectAsmParser &parser) const {
  // Call the default parser for the type.
  Type type;
  auto parseResult = generatedTypeParser(parser, nullptr, type);
  if (parseResult.has_value())
    return type;
  return Type();
}

void ASMDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Call the default printer for the type.
  if (failed(generatedTypePrinter(type, printer)))
    printer << "<unknown asm type>";
}