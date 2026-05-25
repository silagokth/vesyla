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

LogicalResult CstrOp::verify() {
  auto src = getSrc();
  if (src.getIdxLo().size() != src.getIdxHi().size()) {
    return emitOpError("src idx_lo and idx_hi must have the same size");
  }
  if (src.getEvent().empty() != src.getIdxLo().empty()) {
    return emitOpError("src must have both event and indices, or neither");
  }
  for (std::size_t i = 0; i < src.getIdxLo().size(); ++i) {
    if (src.getIdxLo()[i] > src.getIdxHi()[i]) {
      return emitOpError("src idx_lo must be <= idx_hi in every dimension");
    }
  }
  auto dst = getDst();
  if (dst.getIdxLo().size() != dst.getIdxHi().size()) {
    return emitOpError("dst idx_lo and idx_hi must have the same size");
  }
  if (dst.getEvent().empty() != dst.getIdxLo().empty()) {
    return emitOpError("dst must have both event and indices, or neither");
  }
  for (std::size_t i = 0; i < dst.getIdxLo().size(); ++i) {
    if (dst.getIdxLo()[i] > dst.getIdxHi()[i]) {
      return emitOpError("dst idx_lo must be <= idx_hi in every dimension");
    }
  }
  uint64_t src_count = 1;
  for (std::size_t i = 0; i < src.getIdxLo().size(); ++i) {
    src_count *=
        static_cast<uint64_t>(src.getIdxHi()[i] - src.getIdxLo()[i] + 1);
  }
  uint64_t dst_count = 1;
  for (std::size_t i = 0; i < dst.getIdxLo().size(); ++i) {
    dst_count *=
        static_cast<uint64_t>(dst.getIdxHi()[i] - dst.getIdxLo()[i] + 1);
  }
  if (src_count != dst_count) {
    return emitOpError(
        "src and dst must reference the same number of elements");
  }
  return success();
}

LogicalResult IcDepOp::verify() {
  StringRef kind = getKind();
  Attribute src_attr = getSrc();
  Attribute dst_attr = getDst();
  std::optional<StringRef> dir = getDir();

  if (kind == "word") {
    if (dir.has_value()) {
      return emitOpError("kind \"word\" must not carry a dir attribute");
    }
    if (!mlir::isa<ResourceAttr>(src_attr)) {
      return emitOpError("kind \"word\" requires a resource src");
    }
    auto dst = mlir::dyn_cast<ArrayAttr>(dst_attr);
    if (!dst) {
      return emitOpError("kind \"word\" requires a resource-array dst");
    }
    if (dst.empty()) {
      return emitOpError("dst must contain at least one receiver");
    }
    if (dst.size() > 1) {
      return emitOpError(
          "dst may contain more than one receiver only when kind = \"bulk\"");
    }
    return success();
  }
  if (kind == "bulk") {
    if (!mlir::isa<ResourceAttr>(src_attr)) {
      return emitOpError("kind \"bulk\" requires a resource src");
    }
    auto dst = mlir::dyn_cast<ArrayAttr>(dst_attr);
    if (!dst) {
      return emitOpError("kind \"bulk\" requires a resource-array dst");
    }
    if (dst.empty()) {
      return emitOpError("dst must contain at least one receiver");
    }
    if (!dir.has_value()) {
      return success();
    }
    if (*dir != "send" && *dir != "recv") {
      return emitOpError("bulk dir must be one of send|recv, got \"")
             << *dir << "\"";
    }
    if (dst.size() != 1) {
      return emitOpError("bulk dir \"") << *dir
             << "\" dst must contain exactly one receiver";
    }
    return success();
  }
  return emitOpError("kind must be one of word|bulk, got \"") << kind << "\"";
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
