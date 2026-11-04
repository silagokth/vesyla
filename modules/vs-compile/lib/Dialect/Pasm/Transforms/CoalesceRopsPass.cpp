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

// Lanes an AGU program can have: the rep lane field reaches lanes 1..3.
constexpr size_t kMaxLanes = 4;

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

// A rep that only sets one lane's step at the last outer level.
bool is_lane_rep(InstrOp rep) {
  auto lane = llvm::dyn_cast_or_null<mlir::IntegerAttr>(rep.getParam().get("lane"));
  return lane && lane.getInt() != 0;
}

// A rop's reps in the order they sit in, innermost loop first.
llvm::SmallVector<InstrOp> reps_of(RopOp rop) {
  llvm::SmallVector<InstrOp> reps;
  for (InstrOp instr : rop.getBody().front().getOps<InstrOp>()) {
    if (instr.getType() == "rep" && !is_lane_rep(instr)) {
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
// A shared loop the rops step through differently gets lane reps (see
// add_lane_steps). Nothing when the group cannot be merged as one: two rops
// sharing a loop the rest do not would need a merge inside the merge, and the
// lane field reaches lanes 1..3 only.
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
      if (rep.getParam().get("step") != head.getParam().get("step") &&
          rops.size() > kMaxLanes) {
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

// After the base rep of each shared level, one rep per lane whose step there
// differs from the head's: `lane` = its event index, iter and delay unused.
void add_lane_steps(llvm::ArrayRef<InstrOp> head_shared,
                    llvm::ArrayRef<llvm::SmallVector<InstrOp>> lane_shared) {
  for (size_t i = 0; i < head_shared.size(); ++i) {
    InstrOp base = head_shared[i];
    mlir::OpBuilder builder(base.getContext());
    builder.setInsertionPointAfter(base);
    for (size_t k = lane_shared.size(); k-- > 0;) {
      InstrOp rep = lane_shared[k][i];
      if (rep.getParam().get("step") == base.getParam().get("step")) {
        continue;
      }
      llvm::SmallVector<mlir::NamedAttribute> params;
      params.push_back(builder.getNamedAttr("iter", builder.getI32IntegerAttr(0)));
      params.push_back(builder.getNamedAttr("delay", builder.getI32IntegerAttr(0)));
      params.push_back(builder.getNamedAttr("step", rep.getParam().get("step")));
      params.push_back(builder.getNamedAttr(
          "lane", builder.getI32IntegerAttr(static_cast<int32_t>(k + 1))));
      if (mlir::Attribute port = base.getParam().get("port")) {
        params.push_back(builder.getNamedAttr("port", port));
      }
      InstrOp::create(builder, base.getLoc(),
                      builder.getStringAttr(base.getId().str() + "_lane" +
                                            std::to_string(k + 1)),
                      builder.getStringAttr("rep"),
                      builder.getDictionaryAttr(params));
    }
  }
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
// The transitions come after every event, in event order: the AGU controller
// opens a lane per evt, takes the reps that follow it before any trans as that
// lane's own, joins lane k to lane k + 1 through the k-th trans, and reads a rep
// after a trans as one around all the lanes -- after which it drops any further
// trans. So a trans written between two events would turn the next event's own
// reps into outer ones. The timing model (the component library's
// get_timing_model) reads the list the same way, as T<..>(T<..>(e0, e1), e2),
// which is the order they run in.
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

// The port a rop's events run on, or nothing for a rop without events.
std::optional<int32_t> event_port(RopOp rop) {
  for (InstrOp instr : rop.getBody().front().getOps<InstrOp>()) {
    if (instr.getType() != "evt") {
      continue;
    }
    if (auto port =
            llvm::dyn_cast_or_null<mlir::IntegerAttr>(instr.getParam().get("port"))) {
      return static_cast<int32_t>(port.getInt());
    }
  }
  return std::nullopt;
}

// The anchor of a rop's last event: its last event, at the last iteration of
// every rep around it, outermost first. Nothing when a rep's count is not a
// number.
std::optional<AnchorRangeAttr> last_event(RopOp rop) {
  uint32_t event = 0;
  uint32_t events = 0;
  llvm::SmallVector<InstrOp> own;    // reps of the last event, innermost first
  llvm::SmallVector<InstrOp> shared; // reps after the transitions
  bool after_trans = false;
  for (InstrOp instr : rop.getBody().front().getOps<InstrOp>()) {
    if (instr.getType() == "evt") {
      event = events++;
      own.clear();
    } else if (instr.getType() == "trans") {
      after_trans = true;
    } else if (instr.getType() == "rep" && !is_lane_rep(instr)) {
      (after_trans ? shared : own).push_back(instr);
    }
  }
  std::vector<uint32_t> ir;
  for (auto *reps : {&shared, &own}) {
    for (auto it = reps->rbegin(); it != reps->rend(); ++it) {
      auto iter =
          llvm::dyn_cast_or_null<mlir::IntegerAttr>(it->getParam().get("iter"));
      if (!iter) {
        return std::nullopt;
      }
      ir.push_back(static_cast<uint32_t>(std::max<int64_t>(iter.getInt(), 1) - 1));
    }
  }
  return AnchorRangeAttr::get(rop.getContext(),
                              mlir::FlatSymbolRefAttr::get(rop.getSymNameAttr()),
                              {}, event, ir, {}, event, ir);
}

// A port runs one AGU program, so rops with events on the same port that were
// not merged run one after the other, in program order: each one's setup comes
// after the last event of the one before (`port_exclusive`; ScheduleEpochPass
// adds the setup lead). Two that share a loop would have to interleave, which
// no ordering allows: an error.
mlir::LogicalResult order_port_users(EpochOp epoch) {
  std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>,
           llvm::SmallVector<RopOp>>
      by_port;
  for (RopOp rop : epoch.getBody().front().getOps<RopOp>()) {
    if (std::optional<int32_t> port = event_port(rop)) {
      by_port[{rop.getRow(), rop.getCol(), rop.getSlot(), *port}].push_back(rop);
    }
  }
  mlir::OpBuilder builder(epoch.getContext());
  builder.setInsertionPoint(epoch.getBody().front().getTerminator());
  for (auto &[key, rops] : by_port) {
    for (size_t k = 1; k < rops.size(); ++k) {
      RopOp a = rops[k - 1];
      RopOp b = rops[k];
      llvm::StringSet<> loops;
      for (InstrOp rep : reps_of(a)) {
        if (!loop_of(rep).empty()) {
          loops.insert(loop_of(rep));
        }
      }
      for (InstrOp rep : reps_of(b)) {
        if (loops.contains(loop_of(rep))) {
          return b.emitError("coalesce-rops: ")
                 << a.getSymName() << " and " << b.getSymName()
                 << " use port " << std::get<3>(key) << " of slot "
                 << std::get<2>(key)
                 << " in the same loop and cannot be merged into one AGU "
                    "program (their repetitions differ); not supported";
        }
      }
      std::optional<AnchorRangeAttr> src = last_event(a);
      if (!src) {
        return a.emitError("coalesce-rops: ")
               << a.getSymName() << " repeats a non-constant number of times";
      }
      auto dst = AnchorRangeAttr::get(
          epoch.getContext(), mlir::FlatSymbolRefAttr::get(b.getSymNameAttr()),
          {}, 0, {}, {}, 0, {});
      auto cstr = CstrOp::create(
          builder, b.getLoc(), *src, dst,
          DelayAttr::get(epoch.getContext(), 2, std::nullopt),
          builder.getBoolAttr(false));
      cstr->setAttr("port_exclusive", builder.getUnitAttr());
      cstr->setAttr("debug", builder.getStrArrayAttr(
                                 {"CoalesceRopsPass", "port-exclusive"}));
    }
  }
  return mlir::success();
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
        llvm::SmallVector<llvm::SmallVector<InstrOp>> lane_shared;
        for (size_t k = 1; k < rops.size(); ++k) {
          // The event, then the reps that are its own, ahead of every
          // transition; the shared ones it carried go with the rop, the
          // head's standing for them.
          llvm::SmallVector<InstrOp> reps = reps_of(rops[k]);
          move_event(head, sole_event(rops[k]));
          for (size_t i = 0; i + *shared < reps.size(); ++i) {
            reps[i]->moveBefore(end);
          }
          lane_shared.emplace_back(reps.end() - *shared, reps.end());
          absorbed[rops[k].getSymName()] =
              Absorbed{head.getSymNameAttr(), static_cast<uint32_t>(k)};
          merged_away.push_back(rops[k]);
        }
        for (size_t k = 1; k < rops.size(); ++k) {
          append_transition(head, port);
        }
        // The shared reps close the rop, after the last transition, so they
        // repeat the whole sequence of events.
        for (size_t i = head_reps.size() - *shared; i < head_reps.size(); ++i) {
          head_reps[i]->moveBefore(end);
        }
        add_lane_steps(llvm::ArrayRef<InstrOp>(head_reps).take_back(*shared),
                       lane_shared);
      }

      if (absorbed.empty()) {
        if (mlir::failed(order_port_users(epoch))) {
          signalPassFailure();
        }
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
      if (mlir::failed(order_port_users(epoch))) {
        signalPassFailure();
      }
    });
  }
};

} // namespace
} // namespace vesyla::pasm
