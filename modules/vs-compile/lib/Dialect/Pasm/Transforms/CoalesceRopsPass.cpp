#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include "vesyla/Dialect/Pasm/Transforms/CoalesceRopsPass.hpp"
#include "vesyla/Support/AguMerge.hpp"
#include "vesyla/Support/Common.hpp"

#include <map>
#include <optional>
#include <tuple>

namespace vesyla::pasm {
#define GEN_PASS_DEF_COALESCEROPSPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Members of a group that can step on their own: the lane field reaches 1..3.
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

// One AGU program from the rops on one port, nested by their loop tree
// (vesyla/Support/AguMerge.hpp) and written postfix, the way the AGU
// controller and the timing model read it: an evt with its own reps per rop, a
// trans joining each further member of a group, then the group's rep (a range
// level over its lanes). Where a member steps through a group's loop
// differently from the first, a lane rep follows (lane = its place in the
// group, which the lane field reaches up to 3).
class Merger {
public:
  explicit Merger(llvm::ArrayRef<RopOp> rops) : rops_(rops) {
    for (RopOp rop : rops) {
      reps_.push_back(reps_of(rop));
      llvm::SmallVector<const void *> chain;
      for (auto it = reps_.back().rbegin(); it != reps_.back().rend(); ++it) {
        chain.push_back(loop_of(*it).empty() ? nullptr : loop_of(*it).data());
      }
      chains_.push_back(chain);
    }
    tree_ = ::vesyla::util::loop_tree(chains_);
  }

  bool ok() const { return tree_ && lanes_fit(0); }

  // The instructions of the merged rop in order; new trans and lane reps are
  // created in `head`.
  llvm::SmallVector<mlir::Operation *> emit(RopOp head, int32_t port) {
    llvm::SmallVector<mlir::Operation *> seq;
    emit_node(0, head, port, seq);
    return seq;
  }

private:
  // Rop k's rep for its loop at chain index d (outermost 0).
  InstrOp rep_at(size_t k, size_t d) const {
    return reps_[k][reps_[k].size() - 1 - d];
  }

  bool lanes_fit(size_t node) const {
    const auto &n = tree_->nodes[node];
    if (node != 0 && n.rops > 1) {
      InstrOp base = rep_at(n.first, n.depth - 1);
      for (size_t j = n.first + 1; j < n.first + n.rops; ++j) {
        if (rep_at(j, n.depth - 1).getParam().get("step") !=
                base.getParam().get("step") &&
            j - n.first >= kMaxLanes) {
          return false;
        }
      }
    }
    for (const auto &c : n.children) {
      if (!c.is_rop && !lanes_fit(c.index)) {
        return false;
      }
    }
    return true;
  }

  // Rop k's event and its own reps, innermost first, from chain index `from`.
  void emit_rop(size_t k, size_t from,
                llvm::SmallVector<mlir::Operation *> &seq) const {
    seq.push_back(sole_event(rops_[k]));
    for (size_t d = chains_[k].size(); d-- > from;) {
      seq.push_back(rep_at(k, d));
    }
  }

  void emit_node(size_t node, RopOp head, int32_t port,
                 llvm::SmallVector<mlir::Operation *> &seq) const {
    const auto &n = tree_->nodes[node];
    if (node != 0 && n.rops == 1) {
      emit_rop(n.first, n.depth - 1, seq);
      return;
    }
    for (const auto &c : n.children) {
      if (c.is_rop) {
        emit_rop(c.index, n.depth, seq);
      } else {
        emit_node(c.index, head, port, seq);
      }
    }
    for (size_t i = 1; i < n.children.size(); ++i) {
      seq.push_back(append_transition(head, port));
    }
    if (node == 0) {
      return;
    }
    InstrOp base = rep_at(n.first, n.depth - 1);
    seq.push_back(base);
    for (size_t j = n.first + 1; j < n.first + n.rops; ++j) {
      InstrOp rep = rep_at(j, n.depth - 1);
      if (rep.getParam().get("step") != base.getParam().get("step")) {
        seq.push_back(lane_rep(base, rep, j - n.first));
      }
    }
  }

  // A rep setting lane `lane`'s step at the level `base` opens; iter and delay
  // unused.
  static mlir::Operation *lane_rep(InstrOp base, InstrOp rep, size_t lane) {
    mlir::OpBuilder builder(base.getContext());
    builder.setInsertionPointAfter(base);
    llvm::SmallVector<mlir::NamedAttribute> params;
    params.push_back(builder.getNamedAttr("iter", builder.getI32IntegerAttr(0)));
    params.push_back(builder.getNamedAttr("delay", builder.getI32IntegerAttr(0)));
    params.push_back(builder.getNamedAttr("step", rep.getParam().get("step")));
    params.push_back(builder.getNamedAttr(
        "lane", builder.getI32IntegerAttr(static_cast<int32_t>(lane))));
    if (mlir::Attribute port = base.getParam().get("port")) {
      params.push_back(builder.getNamedAttr("port", port));
    }
    return InstrOp::create(builder, base.getLoc(),
                           builder.getStringAttr(base.getId().str() + "_lane" +
                                                 std::to_string(lane)),
                           builder.getStringAttr("rep"),
                           builder.getDictionaryAttr(params));
  }

  static mlir::Operation *append_transition(RopOp head, int32_t port);

  llvm::ArrayRef<RopOp> rops_;
  llvm::SmallVector<llvm::SmallVector<InstrOp>> reps_;
  llvm::SmallVector<llvm::SmallVector<const void *>> chains_;
  std::optional<::vesyla::util::LoopTree> tree_;
};

// A new trans at the end of `head`'s body; its delay is a fresh symbol for the
// scheduler.
mlir::Operation *Merger::append_transition(RopOp head, int32_t port) {
  mlir::OpBuilder builder(head.getContext());
  builder.setInsertionPoint(head.getBody().front().getTerminator());

  std::string suffix = ::vesyla::util::Common::gen_random_string(8);
  llvm::SmallVector<mlir::NamedAttribute> params;
  params.push_back(
      builder.getNamedAttr("delay", builder.getStringAttr("t_" + suffix)));
  params.push_back(
      builder.getNamedAttr("port", builder.getI32IntegerAttr(port)));
  return InstrOp::create(builder, head.getLoc(),
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
                              it->second.event, anchor.getIrHi(),
                              anchor.getIrLoSlope(), anchor.getIrHiSlope(),
                              anchor.getIrFollows());
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
      shared.clear();
      after_trans = false;
    } else if (instr.getType() == "trans") {
      after_trans = true;
    } else if (instr.getType() == "rep" && !is_lane_rep(instr)) {
      (after_trans ? shared : own).push_back(instr);
    }
  }
  // The last index of each rep, outermost first. A rep with a slope (AGU
  // Slope Levels) runs iter + slope * (last index of the rep `follows` reps
  // out) times in the last pass.
  std::vector<uint32_t> ir;
  llvm::SmallVector<int64_t> last;
  for (auto *reps : {&shared, &own}) {
    for (auto it = reps->rbegin(); it != reps->rend(); ++it) {
      auto iter =
          llvm::dyn_cast_or_null<mlir::IntegerAttr>(it->getParam().get("iter"));
      if (!iter) {
        return std::nullopt;
      }
      int64_t trips = iter.getInt();
      auto slope =
          llvm::dyn_cast_or_null<mlir::IntegerAttr>(it->getParam().get("slope"));
      if (slope && slope.getInt() != 0) {
        auto follows_attr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
            it->getParam().get("follows"));
        size_t follows = follows_attr && follows_attr.getInt() > 0
                             ? static_cast<size_t>(follows_attr.getInt())
                             : 1;
        if (follows <= last.size()) {
          trips += slope.getInt() * last[last.size() - follows];
        }
      }
      trips = std::max<int64_t>(trips, 1);
      last.push_back(trips - 1);
      ir.push_back(static_cast<uint32_t>(trips - 1));
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
        Merger merger(rops);
        if (!merger.ok()) {
          continue;
        }
        RopOp head = rops.front();
        int32_t port = std::get<3>(entry.first);
        mlir::Operation *end = head.getBody().front().getTerminator();
        for (mlir::Operation *op : merger.emit(head, port)) {
          op->moveBefore(end);
        }
        for (size_t k = 1; k < rops.size(); ++k) {
          absorbed[rops[k].getSymName()] =
              Absorbed{head.getSymNameAttr(), static_cast<uint32_t>(k)};
          merged_away.push_back(rops[k]);
        }
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
