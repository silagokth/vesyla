#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "vesyla/Dialect/Pasm/Transforms/CoalesceRopsPass.hpp"
#include "vesyla/Support/Common.hpp"

#include <map>
#include <optional>
#include <tuple>

namespace vesyla::pasm {
#define GEN_PASS_DEF_COALESCEROPSPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Where a rop that was merged away ended up: the rop that absorbed it, and
// which of that rop's events it became.
struct Absorbed {
  mlir::StringAttr into;
  uint32_t event;
};

// The one evt of a rop that can join a merge, or null.
//
// A rop qualifies when its body is that one event, followed by nothing but the
// repetitions of the loops it sat in. A conf is positional -- a route or swb
// option is identified by the order it sits in -- so a rop carrying one is
// left alone.
InstrOp sole_event(RopOp rop) {
  InstrOp event;
  for (mlir::Operation &op : rop.getBody().front()) {
    auto instr = llvm::dyn_cast<InstrOp>(op);
    if (!instr) {
      continue;
    }
    if (event && instr.getType() == "rep") {
      continue;
    }
    if (instr.getType() != "evt" || event) {
      return nullptr;
    }
    event = instr;
  }
  return event;
}

// A rop's reps in the order they sit in, innermost loop first.
llvm::SmallVector<InstrOp> reps_of(RopOp rop) {
  llvm::SmallVector<InstrOp> reps;
  for (InstrOp instr : rop.getBody().front().getOps<InstrOp>()) {
    if (instr.getType() == "rep") {
      reps.push_back(instr);
    }
  }
  return reps;
}

// The loop a rep was lowered from, as AffineToRepPass named it; empty for a
// rep that came from somewhere else.
llvm::StringRef loop_of(InstrOp rep) {
  auto tag = rep->getAttrOfType<mlir::StringAttr>("loop");
  return tag ? tag.getValue() : llvm::StringRef();
}

// How many of their outermost reps the rops of a group have in common: the
// loops all of them sat in. Those stay one repetition around the merged
// events, while every other rep is its event's own and stays inside the
// transition with it.
//
// Nothing when the group cannot be merged as one: a shared loop the rops step
// through differently has no single repetition that serves them all, and two
// rops sharing a loop the rest do not would need a merge inside the merge.
std::optional<size_t> shared_reps(llvm::ArrayRef<RopOp> rops) {
  llvm::SmallVector<llvm::SmallVector<InstrOp>> reps;
  for (RopOp rop : rops) {
    reps.push_back(reps_of(rop));
  }

  size_t shared = 0;
  while (true) {
    bool all = true;
    for (const auto &own : reps) {
      if (own.size() <= shared) {
        all = false;
        break;
      }
      InstrOp rep = own[own.size() - 1 - shared];
      InstrOp head = reps.front()[reps.front().size() - 1 - shared];
      if (loop_of(rep).empty() || loop_of(rep) != loop_of(head)) {
        all = false;
        break;
      }
      if (rep.getParam().get("step") != head.getParam().get("step")) {
        return std::nullopt;
      }
    }
    if (!all) {
      break;
    }
    ++shared;
  }

  llvm::StringSet<> seen;
  for (const auto &own : reps) {
    for (size_t i = 0; i + shared < own.size(); ++i) {
      if (!loop_of(own[i]).empty() && !seen.insert(loop_of(own[i])).second) {
        return std::nullopt;
      }
    }
  }
  return shared;
}

// Move `event` to the end of `head`'s body.
//
// A resource asked for two things in succession wants one instruction stream
// with a transition joining them, not two instructions: on a second
// instruction it retriggers, and the transfer the first one started is lost.
// The events keep their own parameters -- each addresses wherever it addressed
// before.
void move_event(RopOp head, InstrOp event) {
  event->moveBefore(head.getBody().front().getTerminator());
}

// Add one transition to the end of `head`'s body.
//
// A transition follows the event it joins to the one before it -- after the
// second evt, then after the third, and so on. The resource reads its own
// instruction list as a postfix stack when it works out the timing model: an
// evt pushes, a trans pops the two below it and combines them. So a trans
// written between two events finds only one on the stack, is silently
// declined, and the operation loses an event; written after the second it
// closes the pair. Joining them as it goes leaves the expression nested to the
// left, T<..>(T<..>(e0, e1), e2), which is the order they run in.
//
// The delay is a fresh symbol; how far apart the two events sit is the
// scheduler's to decide, and the constraints that used to order the separate
// rops now order the events.
void append_transition(RopOp head, int32_t port) {
  mlir::OpBuilder builder(head.getContext());
  builder.setInsertionPoint(head.getBody().front().getTerminator());

  std::string suffix = ::vesyla::util::Common::gen_random_string(8);
  llvm::SmallVector<mlir::NamedAttribute> params;
  params.push_back(
      builder.getNamedAttr("delay", builder.getStringAttr("t_" + suffix)));
  params.push_back(
      builder.getNamedAttr("port", builder.getI32IntegerAttr(port)));
  InstrOp::create(builder, head.getLoc(),
                  builder.getStringAttr(head.getSymName().str() + "_" + suffix),
                  builder.getStringAttr("trans"),
                  builder.getDictionaryAttr(params));
}

// The same anchor, re-pointed at the event it became.
//
// Only the event id moves. The indices say which repeat of the event is meant,
// outermost first, and the merged event sits under the same repetitions in the
// same order as it did in its own rop -- the shared ones are now around the
// transition rather than directly around the event, which the timing model
// reads off the rop and not off the anchor.
AnchorRangeAttr remap(AnchorRangeAttr anchor,
                      const llvm::StringMap<Absorbed> &absorbed) {
  auto it = absorbed.find(anchor.getInstr().getValue());
  if (it == absorbed.end()) {
    return anchor;
  }
  return AnchorRangeAttr::get(anchor.getContext(),
                              mlir::FlatSymbolRefAttr::get(it->second.into),
                              anchor.getOrLo(), it->second.event,
                              anchor.getIrLo(), anchor.getOrHi(),
                              it->second.event, anchor.getIrHi());
}

AnchorAttr remap(AnchorAttr anchor, const llvm::StringMap<Absorbed> &absorbed) {
  auto it = absorbed.find(anchor.getInstr().getValue());
  if (it == absorbed.end()) {
    return anchor;
  }
  return AnchorAttr::get(anchor.getContext(),
                         mlir::FlatSymbolRefAttr::get(it->second.into),
                         anchor.getOrIdx(),
                         static_cast<int32_t>(it->second.event),
                         anchor.getIrIdx(), anchor.getDelay());
}

class CoalesceRopsPass : public impl::CoalesceRopsPassBase<CoalesceRopsPass> {
public:
  using impl::CoalesceRopsPassBase<CoalesceRopsPass>::CoalesceRopsPassBase;

  // Walked directly rather than driven by a rewrite pattern: merging rewrites
  // anchors on operations elsewhere in the epoch, which is not a local
  // rewrite, and the greedy driver has nothing to converge on once the merged
  // rops are gone.
  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    module.walk([&](EpochOp epoch) {
      llvm::StringMap<Absorbed> absorbed;
      llvm::SmallVector<RopOp> merged_away;

      // Resource-centric: gather the events on each (row, col, slot, port) in
      // program order. A transition applies to one port, so the port is part
      // of the identity; what sits between them on other resources is another
      // instruction on other hardware and has no bearing on this one.
      std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>,
               llvm::SmallVector<RopOp>>
          by_resource;
      for (RopOp rop : epoch.getBody().front().getOps<RopOp>()) {
        InstrOp event = sole_event(rop);
        if (!event) {
          continue;
        }
        auto port = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
            event.getParam().get("port"));
        if (!port) {
          continue;
        }
        by_resource[{rop.getRow(), rop.getCol(), rop.getSlot(),
                     static_cast<int32_t>(port.getInt())}]
            .push_back(rop);
      }

      for (auto &entry : by_resource) {
        llvm::SmallVector<RopOp> &rops = entry.second;
        if (rops.size() < 2) {
          continue;
        }
        std::optional<size_t> shared = shared_reps(rops);
        if (!shared) {
          continue;
        }
        RopOp head = rops.front();
        int32_t port = std::get<3>(entry.first);
        mlir::Operation *end = head.getBody().front().getTerminator();
        llvm::SmallVector<InstrOp> head_reps = reps_of(head);
        for (size_t k = 1; k < rops.size(); ++k) {
          // The event, then the reps that are its own; the shared ones it
          // carried go with the rop, the head's standing for them.
          llvm::SmallVector<InstrOp> reps = reps_of(rops[k]);
          move_event(head, sole_event(rops[k]));
          for (size_t i = 0; i + *shared < reps.size(); ++i) {
            reps[i]->moveBefore(end);
          }
          append_transition(head, port);
          absorbed[rops[k].getSymName()] =
              Absorbed{head.getSymNameAttr(), static_cast<uint32_t>(k)};
          merged_away.push_back(rops[k]);
        }
        // The shared reps close the rop, after the last transition, so they
        // repeat the whole sequence of events.
        for (size_t i = head_reps.size() - *shared; i < head_reps.size(); ++i) {
          head_reps[i]->moveBefore(end);
        }
      }

      if (absorbed.empty()) {
        return;
      }

      // Re-anchor everything that named a rop which is now an event of
      // another. The interconnect dependencies matter as much as the
      // constraints: the interconnect pass reads them after this one.
      //
      // A constraint whose two ends have become one rop is kept. Its ends are
      // still two different events, which is what the transition between them
      // is there to space, and both the timing model and the interconnect walk
      // tell events apart by their id.
      epoch.walk([&](CstrOp cstr) {
        cstr.setSrcAttr(remap(cstr.getSrc(), absorbed));
        cstr.setDstAttr(remap(cstr.getDst(), absorbed));
      });

      epoch.walk([&](IcDepOp icdep) {
        icdep.setFirstAttr(remap(icdep.getFirst(), absorbed));
        icdep.setLastAttr(remap(icdep.getLast(), absorbed));
      });

      for (RopOp rop : merged_away) {
        rop.erase();
      }
    });
  }
};

} // namespace
} // namespace vesyla::pasm
