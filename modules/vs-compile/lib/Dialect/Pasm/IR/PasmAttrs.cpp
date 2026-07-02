#include "vesyla/Dialect/Pasm/IR/PasmAttrs.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "vesyla/Dialect/Pasm/IR/PasmDialect.hpp"
#include "vesyla/Support/Anchor.hpp"
#include "llvm/ADT/TypeSwitch.h"

using namespace vesyla::pasm;

#define GET_ATTRDEF_CLASSES
#include "vesyla/Dialect/Pasm/IR/PasmAttrs.cpp.inc"

void PasmDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "vesyla/Dialect/Pasm/IR/PasmAttrs.cpp.inc"
      >();
}

mlir::Attribute ResourceAttr::parse(mlir::AsmParser &p, mlir::Type) {
  if (p.parseLess()) {
    return {};
  }
  int32_t row = 0, col = 0, slot = 0, port = 0;

  auto assign_field = [&](llvm::StringRef k, int32_t v) -> mlir::LogicalResult {
    if (k == "row") {
      row = v;
    } else if (k == "col") {
      col = v;
    } else if (k == "slot") {
      slot = v;
    } else if (k == "port") {
      port = v;
    } else {
      return mlir::failure();
    }
    return mlir::success();
  };

  // Accept both the verbose "row = .., col = .., .." form and the short
  // positional "<row, col, slot, port>" form.
  llvm::StringRef key;
  if (mlir::succeeded(p.parseOptionalKeyword(&key))) {
    int32_t v;
    if (p.parseEqual() || p.parseInteger(v) || mlir::failed(assign_field(key, v))) {
      return {};
    }
    while (mlir::succeeded(p.parseOptionalComma())) {
      llvm::StringRef k;
      if (p.parseKeyword(&k) || p.parseEqual() || p.parseInteger(v) ||
          mlir::failed(assign_field(k, v))) {
        return {};
      }
    }
  } else {
    if (p.parseInteger(row) || p.parseComma() || p.parseInteger(col) ||
        p.parseComma() || p.parseInteger(slot) || p.parseComma() ||
        p.parseInteger(port)) {
      return {};
    }
  }

  if (p.parseGreater()) {
    return {};
  }
  return ResourceAttr::get(p.getContext(), row, col, slot, port);
}

void ResourceAttr::print(mlir::AsmPrinter &p) const {
  p << "<" << getRow() << ", " << getCol() << ", " << getSlot() << ", "
    << getPort() << ">";
}

mlir::Attribute AnchorRangeAttr::parse(mlir::AsmParser &p, mlir::Type) {
  if (p.parseLess()) {
    return {};
  }

  mlir::StringAttr instr_name;
  if (p.parseSymbolName(instr_name)) {
    return {};
  }
  auto instr = mlir::FlatSymbolRefAttr::get(p.getContext(), instr_name);

  // The index text (OR.MT.IR with optional range) is carried as a quoted
  // string so it round-trips through the shared ::vesyla::Anchor grammar.
  std::string idx_text;
  if (mlir::succeeded(p.parseOptionalComma())) {
    if (p.parseString(&idx_text)) {
      return {};
    }
  }

  if (p.parseGreater()) {
    return {};
  }

  auto range = ::vesyla::AnchorRange::parse(instr_name.getValue().str() + idx_text);
  if (!range) {
    p.emitError(p.getCurrentLocation(), "invalid anchor_range index text: ")
        << idx_text;
    return {};
  }

  auto to_u = [](const std::vector<int> &v) {
    return llvm::SmallVector<uint32_t>(v.begin(), v.end());
  };
  return AnchorRangeAttr::get(
      p.getContext(), instr, to_u(range->lo.or_idx),
      static_cast<uint32_t>(range->lo.mt_idx), to_u(range->lo.ir_idx),
      to_u(range->hi.or_idx), static_cast<uint32_t>(range->hi.mt_idx),
      to_u(range->hi.ir_idx));
}

void AnchorRangeAttr::print(mlir::AsmPrinter &p) const {
  p << "<";
  p.printAttribute(getInstr());

  ::vesyla::Anchor lo, hi;
  lo.or_idx.assign(getOrLo().begin(), getOrLo().end());
  lo.mt_idx = static_cast<int>(getMtLo());
  lo.ir_idx.assign(getIrLo().begin(), getIrLo().end());
  hi.or_idx.assign(getOrHi().begin(), getOrHi().end());
  hi.mt_idx = static_cast<int>(getMtHi());
  hi.ir_idx.assign(getIrHi().begin(), getIrHi().end());

  // Empty-name to_string yields just the index text ("" when fully bare).
  std::string idx = ::vesyla::AnchorRange{lo, hi}.to_string();
  if (!idx.empty()) {
    p << ", \"" << idx << "\"";
  }
  p << ">";
}

mlir::Attribute DelayAttr::parse(mlir::AsmParser &p, mlir::Type) {
  if (p.parseLess() || p.parseLSquare()) {
    return {};
  }

  std::optional<int32_t> min_v;
  std::optional<int32_t> max_v;

  // First slot: either an int followed by ',', or just ',' for an empty min.
  if (p.parseOptionalComma()) {
    int32_t v;
    if (p.parseInteger(v) || p.parseComma()) {
      return {};
    }
    min_v = v;
  }

  // Second slot: either an int followed by ']', or just ']' for an empty max.
  if (p.parseOptionalRSquare()) {
    int32_t v;
    if (p.parseInteger(v) || p.parseRSquare()) {
      return {};
    }
    max_v = v;
  }

  if (p.parseGreater()) {
    return {};
  }
  return DelayAttr::get(p.getContext(), min_v, max_v);
}

void DelayAttr::print(mlir::AsmPrinter &p) const {
  p << "<[";
  if (getMin()) {
    p << *getMin();
  }
  p << ", ";
  if (getMax()) {
    p << *getMax();
  }
  p << "]>";
}

mlir::Attribute AnchorAttr::parse(mlir::AsmParser &p, mlir::Type) {
  if (p.parseLess()) {
    return {};
  }

  mlir::FlatSymbolRefAttr instr;
  if (p.parseAttribute(instr)) {
    return {};
  }

  // Optional trailing parts, each preceded by ','.
  // Forms:  , "<index-text>"  |  , delay  |  , "<index-text>", delay
  std::string idx_text;
  int32_t delay = 0;
  if (mlir::succeeded(p.parseOptionalComma())) {
    if (mlir::succeeded(p.parseOptionalString(&idx_text))) {
      if (mlir::succeeded(p.parseOptionalComma())) {
        if (p.parseInteger(delay)) {
          return {};
        }
      }
    } else if (p.parseInteger(delay)) {
      return {};
    }
  }

  if (p.parseGreater()) {
    return {};
  }

  auto anc = ::vesyla::Anchor::parse(instr.getValue().str() + idx_text);
  if (!anc) {
    p.emitError(p.getCurrentLocation(), "invalid anchor index text: ")
        << idx_text;
    return {};
  }
  auto to_i = [](const std::vector<int> &v) {
    return llvm::SmallVector<int32_t>(v.begin(), v.end());
  };
  return AnchorAttr::get(p.getContext(), instr, to_i(anc->or_idx),
                         static_cast<int32_t>(anc->mt_idx), to_i(anc->ir_idx),
                         delay);
}

void AnchorAttr::print(mlir::AsmPrinter &p) const {
  p << "<";
  p.printAttribute(getInstr());

  ::vesyla::Anchor anc;
  anc.or_idx.assign(getOrIdx().begin(), getOrIdx().end());
  anc.mt_idx = static_cast<int>(getMt());
  anc.ir_idx.assign(getIrIdx().begin(), getIrIdx().end());

  std::string idx = anc.to_string(); // empty name -> index text ("" if bare)
  if (!idx.empty()) {
    p << ", \"" << idx << "\"";
  }
  if (getDelay() != 0) {
    p << ", " << getDelay();
  }
  p << ">";
}
