#ifndef VESYLA_PASM_LOOP_LEVEL_DETAIL_HPP
#define VESYLA_PASM_LOOP_LEVEL_DETAIL_HPP

#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp"
#include "vesyla/Support/RandName.hpp"

#include <cstdint>
#include <utility>

// Shared by ReplaceLoopPass and ExpandEvtStridesPass. A "loop level" is a lane
// of the loop_var bus the sequencer broadcasts: level 0 = outermost loop.
namespace vesyla::pasm::loop_level_detail {

// evt.stride is 8-bit; a wider stride must ride on an evts.
constexpr int64_t EVT_STRIDE_MAX = 255;
// "No explicit loop_level" (the field's max, filled by AddDefaultValuePass).
constexpr int LOOP_LEVEL_AUTO = 3;

// Valid only before ReplaceLoopOp unwraps the loops.
inline int enclosing_loops(mlir::Operation *op) {
  int n = 0;
  for (mlir::Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (llvm::isa<LoopOp>(p)) {
      ++n;
    }
  }
  return n;
}

// Lane of the innermost enclosing loop; 0 if there is none.
inline int innermost_loop_level(mlir::Operation *op) {
  int n = enclosing_loops(op);
  return n > 0 ? n - 1 : 0;
}

// Drop the `remove` params, set the given integer ones, keep the rest.
inline void
update_params(InstrOp op, mlir::OpBuilder &b,
              llvm::ArrayRef<llvm::StringRef> remove,
              llvm::ArrayRef<std::pair<llvm::StringRef, int64_t>> set) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  for (mlir::NamedAttribute na : op.getParam()) {
    llvm::StringRef n = na.getName();
    bool overwritten =
        llvm::any_of(set, [&](const auto &kv) { return kv.first == n; });
    if (!llvm::is_contained(remove, n) && !overwritten) {
      attrs.push_back(na);
    }
  }
  for (const auto &kv : set) {
    attrs.push_back(b.getNamedAttr(kv.first, b.getI32IntegerAttr(kv.second)));
  }
  op->setAttr("param", b.getDictionaryAttr(attrs));
}

// Emit `evts` at b's insertion point, inheriting the evt's slot/port.
inline void emit_evts(mlir::OpBuilder &b, InstrOp evt, int64_t stride,
                      int level) {
  llvm::SmallVector<mlir::NamedAttribute> params;
  for (llvm::StringRef name : {"slot", "port"}) {
    if (mlir::Attribute a = evt.getParam().get(name)) {
      params.push_back(b.getNamedAttr(name, a));
    }
  }
  params.push_back(b.getNamedAttr("stride", b.getI32IntegerAttr(stride)));
  params.push_back(b.getNamedAttr("loop_level", b.getI32IntegerAttr(level)));
  b.create<InstrOp>(evt.getLoc(), b.getStringAttr(util::RandName::generate(8)),
                    b.getStringAttr("evts"), b.getDictionaryAttr(params));
}

} // namespace vesyla::pasm::loop_level_detail

#endif // VESYLA_PASM_LOOP_LEVEL_DETAIL_HPP
