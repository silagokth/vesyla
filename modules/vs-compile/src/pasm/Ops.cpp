//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pasm/Ops.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "pasm/Dialect.hpp"
#include "pasm/Types.hpp"

using namespace mlir;
using namespace vesyla::pasm;

#define GET_OP_CLASSES
#include "pasm/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

void PasmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "pasm/Ops.cpp.inc"
      >();
}

// void MakeInstrOp::build(OpBuilder &builder, OperationState &state, StringRef
// id,
//                         StringRef type, DictionaryAttr param, Type
//                         resultType) {
//   state.addAttribute("id", builder.getStringAttr(id));
//   state.addAttribute("type", builder.getStringAttr(type));
//   state.addAttribute("param", param);
//   state.addTypes(resultType);
// }

// ParseResult MakeInstrOp::parse(OpAsmParser &parser, OperationState &result) {
//   // Parse the string attribute
//   StringAttr idAttr;
//   if (parser.parseAttribute(idAttr, "id", result.attributes))
//     return failure();

//   // Parse the dictionary attribute
//   DictionaryAttr paramAttr;
//   if (parser.parseAttribute(paramAttr, "param", result.attributes))
//     return failure();

//   // Parse the result type
//   Type resultType;
//   if (parser.parseColonType(resultType))
//     return failure();
//   result.addTypes(resultType);

//   return success();
// }

// void MakeInstrOp::print(OpAsmPrinter &p) {
//   p << getOperationName() << ' ';
//   p.printAttributeWithoutType(getIdAttr());
//   p << ", ";
//   p.printAttributeWithoutType(getParamAttr());
//   p << " : ";
//   p.printType(getResult().getType());
// }