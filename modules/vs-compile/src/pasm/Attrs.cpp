#include "pasm/Attrs.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "pasm/Dialect.hpp"
#include "llvm/ADT/TypeSwitch.h"

using namespace vesyla::pasm;

#define GET_ATTRDEF_CLASSES
#include "pasm/Attrs.cpp.inc"

void PasmDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "pasm/Attrs.cpp.inc"
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

  std::string event;
  llvm::SmallVector<uint32_t> los;
  llvm::SmallVector<uint32_t> his;

  if (mlir::succeeded(p.parseOptionalComma())) {
    if (p.parseString(&event)) {
      return {};
    }
    if (mlir::succeeded(p.parseOptionalComma())) {
      while (mlir::succeeded(p.parseOptionalLSquare())) {
        uint32_t lo_v;
        if (p.parseInteger(lo_v)) {
          return {};
        }
        uint32_t hi_v = lo_v;
        if (mlir::succeeded(p.parseOptionalColon())) {
          if (p.parseInteger(hi_v)) {
            return {};
          }
        }
        if (p.parseRSquare()) {
          return {};
        }
        los.push_back(lo_v);
        his.push_back(hi_v);
      }
    }
  }

  if (p.parseGreater()) {
    return {};
  }
  return AnchorRangeAttr::get(p.getContext(), instr, event, los, his);
}

void AnchorRangeAttr::print(mlir::AsmPrinter &p) const {
  p << "<";
  p.printAttribute(getInstr());

  llvm::StringRef event = getEvent();
  auto lo = getIdxLo();
  auto hi = getIdxHi();

  if (!event.empty() || !lo.empty()) {
    p << ", \"" << event << "\"";
  }
  if (!lo.empty()) {
    p << ", ";
    for (std::size_t i = 0; i < lo.size(); ++i) {
      p << "[" << lo[i];
      if (lo[i] != hi[i]) {
        p << ":" << hi[i];
      }
      p << "]";
    }
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

  std::string event;
  llvm::SmallVector<int32_t> idx;
  int32_t delay = 0;

  // Optional trailing parts, each preceded by ','.
  // Forms:
  //   , "event", [idx0, idx1, ...]
  //   , delay
  //   , "event", [idx0, ...], delay
  if (mlir::succeeded(p.parseOptionalComma())) {
    std::string ev;
    if (mlir::succeeded(p.parseOptionalString(&ev))) {
      event = ev;
      if (p.parseComma() || p.parseLSquare()) {
        return {};
      }
      if (mlir::failed(p.parseOptionalRSquare())) {
        int32_t v;
        if (p.parseInteger(v)) {
          return {};
        }
        idx.push_back(v);
        while (mlir::succeeded(p.parseOptionalComma())) {
          if (p.parseInteger(v)) {
            return {};
          }
          idx.push_back(v);
        }
        if (p.parseRSquare()) {
          return {};
        }
      }
      if (mlir::succeeded(p.parseOptionalComma())) {
        if (p.parseInteger(delay)) {
          return {};
        }
      }
    } else {
      if (p.parseInteger(delay)) {
        return {};
      }
    }
  }

  if (p.parseGreater()) {
    return {};
  }
  return AnchorAttr::get(p.getContext(), instr, event, idx, delay);
}

void AnchorAttr::print(mlir::AsmPrinter &p) const {
  p << "<";
  p.printAttribute(getInstr());

  bool has_event = !getEvent().empty();
  bool has_delay = getDelay() != 0;

  if (has_event) {
    p << ", \"" << getEvent() << "\", [";
    auto idx = getIdx();
    for (std::size_t i = 0; i < idx.size(); ++i) {
      if (i > 0) {
        p << ", ";
      }
      p << idx[i];
    }
    p << "]";
  }
  if (has_delay) {
    p << ", " << getDelay();
  }
  p << ">";
}
