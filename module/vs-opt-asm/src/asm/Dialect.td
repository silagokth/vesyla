//===- Dialect.td - ASM dialect definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the ASM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASM_DIALECT
#define ASM_DIALECT

include "mlir/IR/DialectBase.td"

// Provide a definition of the 'asm' dialect to define the operations.
def ASM_Dialect : Dialect {
  let name = "asm";
  let summary = "A dialect to represent the assembly language of VESYLA DRRA2.";
  let description = [{
      A dialect to represent the assembly language of VESYLA DRRA2.
      At this level of abstraction, the operations are scheduled and mapped, ready to generate the final assembly.
  }];
  let cppNamespace = "vesyla::asmd";  // Changed from "vesyla::asm" to "vesyla::asmd"

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerOps();
    void registerTypes();
    
    // mlir::Type parseType(mlir::DialectAsmParser &parser) const;
    // void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const;
  }];
}

#endif // ASM_DIALECT