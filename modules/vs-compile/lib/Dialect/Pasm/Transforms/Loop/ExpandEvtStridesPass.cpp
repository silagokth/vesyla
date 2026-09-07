#include "mlir/IR/BuiltinOps.h"

#include "vesyla/Dialect/Pasm/Transforms/ExpandEvtStridesPass.hpp"
#include "vesyla/Support/Common.hpp"

#include "LoopLevelDetail.hpp"

#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>

namespace vesyla::pasm {
#define GEN_PASS_DEF_EXPANDEVTSTRIDESPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {
using namespace loop_level_detail;

using Term = std::pair<int, int64_t>; // (loop level, stride)

// Parse `strides="s0,s1,.."`: index k is loop level k, a 0 entry skips it.
bool parse_strides(InstrOp evt, const std::string &text,
                   llvm::SmallVectorImpl<Term> &terms) {
  int depth = enclosing_loops(evt);
  std::stringstream ss(text);
  std::string tok;
  for (int level = 0; std::getline(ss, tok, ','); ++level) {
    const char *begin = tok.c_str();
    char *end = nullptr;
    int64_t stride = std::strtoll(begin, &end, 10);
    if (end == begin || stride == 0) {
      continue;
    }
    if (level >= depth) {
      llvm::outs() << "Error: evt strides index " << level
                   << " has no enclosing loop at that depth (only " << depth
                   << " enclosing loop(s)).\n";
      return false;
    }
    terms.push_back({level, stride});
  }
  return true;
}

// Spreads an evt's per-iteration address offset over evt + evts terms. The
// offset is a sum of stride * loop_var[level]; the evt carries one term, each
// extra one becomes an evts. Terms come either from a `strides` vector or from
// the evt's own `stride` (which only needs an evts if it overflows 8 bits).
//
// Must run before ScheduleEpoch: its JSON round-trip int-izes string attrs, so
// the `strides` string has to be consumed first.
class ExpandEvtStridesPass
    : public impl::ExpandEvtStridesPassBase<ExpandEvtStridesPass> {
public:
  using impl::ExpandEvtStridesPassBase<
      ExpandEvtStridesPass>::ExpandEvtStridesPassBase;
  void runOnOperation() final {
    mlir::OpBuilder b(&getContext());
    llvm::SmallVector<InstrOp> evts;
    getOperation()->walk([&](InstrOp op) {
      if (op.getType().str() == "evt") {
        evts.push_back(op);
      }
    });

    for (InstrOp evt : evts) {
      llvm::SmallVector<Term> terms;
      if (auto sv = llvm::dyn_cast_or_null<mlir::StringAttr>(
              evt.getParam().get("strides"))) {
        if (!parse_strides(evt, sv.str(), terms)) {
          signalPassFailure();
          return;
        }
      } else {
        auto sa = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
            evt.getParam().get("stride"));
        int64_t stride = sa ? sa.getInt() : 0;
        if (stride <= EVT_STRIDE_MAX) {
          continue; // fits on the evt as written
        }
        auto lv = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
            evt.getParam().get("loop_level"));
        int level = (lv && lv.getInt() != LOOP_LEVEL_AUTO)
                        ? lv.getInt()
                        : innermost_loop_level(evt);
        terms.push_back({level, stride});
      }

      // The evt takes the first term that fits its 8-bit stride; the rest, and
      // any term too wide for it, become evts.
      int on_evt = -1;
      Term term0{terms.empty() ? 0 : terms.front().first, 0};
      for (size_t i = 0; i < terms.size(); ++i) {
        if (terms[i].second <= EVT_STRIDE_MAX) {
          on_evt = static_cast<int>(i);
          term0 = terms[i];
          break;
        }
      }
      update_params(evt, b, /*remove=*/{"strides"},
                    {{"stride", term0.second}, {"loop_level", term0.first}});

      b.setInsertionPointAfter(evt);
      for (size_t i = 0; i < terms.size(); ++i) {
        if (static_cast<int>(i) != on_evt) {
          emit_evts(b, evt, terms[i].second, terms[i].first);
        }
      }
    }
  }
};

} // namespace
} // namespace vesyla::pasm
