//===- Ops.td - ASM dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the ASM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASM_OPS
#define ASM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

include "asm/Dialect.td"
include "asm/Types.td"

// Base class for asm dialect operations. This operation inherits from the base
// `Op` class in OpBase.td and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the prefix.
//   * A list of traits for the operation.
class ASM_Op<string mnemonic, list<Trait> traits = []> :
    Op<ASM_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// ASM operations
//===----------------------------------------------------------------------===//

def DSUOp : ASM_Op<"dsu"> {
    let summary = "Data Storage Unit operation";
    let description = [{
        The DSU operation is used to define a data movement operation for the
        target resource.
    }];

    let arguments = (ins I32Attr:$init_addr_sd);
    let results = (outs InstrType:$result);

    let builders = [
        OpBuilder<(ins "IntegerAttr":$init_addr_sd), [{
            build($_builder, $_state, init_addr_sd);
        }]>
    ];
}

#endif // ASM_OPS%
