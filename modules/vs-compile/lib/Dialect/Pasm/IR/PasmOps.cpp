//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmTypes.hpp"

using namespace mlir;
using namespace vesyla::pasm;

#define GET_OP_CLASSES
#include "vesyla/Dialect/Pasm/IR/PasmOps.cpp.inc"

//===----------------------------------------------------------------------===//

void PasmDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "vesyla/Dialect/Pasm/IR/PasmOps.cpp.inc"
      >();
}

LogicalResult CstrOp::verify() {
  // A range spans only IR indices; OR and MT are shared between lo and hi.
  auto check = [&](AnchorRangeAttr a, const char *which) -> LogicalResult {
    if (a.getOrLo() != a.getOrHi() || a.getMtLo() != a.getMtHi()) {
      return emitOpError() << which << " range may span only IR indices";
    }
    if (a.getIrLo().size() != a.getIrHi().size()) {
      return emitOpError() << which << " ir_lo and ir_hi must have equal size";
    }
    for (std::size_t i = 0; i < a.getIrLo().size(); ++i) {
      if (a.getIrLo()[i] > a.getIrHi()[i]) {
        return emitOpError()
               << which << " ir_lo must be <= ir_hi in every dimension";
      }
    }
    return success();
  };
  if (failed(check(getSrc(), "src")) || failed(check(getDst(), "dst"))) {
    return failure();
  }

  // Total elements = product of the per-dimension IR spans.
  auto count = [](AnchorRangeAttr a) -> uint64_t {
    uint64_t n = 1;
    for (std::size_t i = 0; i < a.getIrLo().size(); ++i) {
      n *= static_cast<uint64_t>(a.getIrHi()[i] - a.getIrLo()[i] + 1);
    }
    return n;
  };
  if (count(getSrc()) != count(getDst())) {
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
