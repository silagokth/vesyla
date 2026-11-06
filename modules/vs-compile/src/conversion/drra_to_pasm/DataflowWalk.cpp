#include "DataflowWalk.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

#include <algorithm>

namespace vesyla::conversion::drra_to_pasm {

namespace {

// The delay accumulated so far and the operation to continue from: a drra.rop,
// or an affine.for entered through an iter_args init. `ii` is the largest
// initiation interval met since the chain last left a loop.
struct WorkItem {
  int32_t delay;
  int32_t ii;
  mlir::Operation *operation;
};

// Cycles from the event that starts one pass to its result; `delay` is the
// older name.
int32_t rop_latency(mlir::Operation *op) {
  for (llvm::StringRef name : {"latency", "delay"}) {
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name)) {
      return static_cast<int32_t>(attr.getInt());
    }
  }
  return 0;
}

// Cycles between two passes; without one a pass occupies the whole latency.
int32_t rop_ii(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("ii")) {
    return static_cast<int32_t>(attr.getInt());
  }
  return rop_latency(op);
}

int64_t trip_count(mlir::affine::AffineForOp loop) {
  int64_t lb = loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
  int64_t ub = loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 0;
  int64_t step = loop.getStepAsInt();
  return step == 0 ? 0 : (ub - lb) / step;
}

} // namespace

llvm::SmallVector<ReachedEvent> reached_events(mlir::Operation *source) {
  llvm::SmallVector<ReachedEvent> reached;
  llvm::SmallVector<WorkItem> working_set{{0, 0, source}};
  while (!working_set.empty()) {
    WorkItem item = working_set.pop_back_val();
    int32_t delay = item.delay + rop_latency(item.operation);
    int32_t ii = std::max(item.ii, rop_ii(item.operation));

    struct Source {
      mlir::Value value;
      int32_t delay;
      int32_t ii;
    };
    llvm::SmallVector<Source> sources;
    if (auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(item.operation)) {
      for (mlir::BlockArgument iter_arg : loop.getRegionIterArgs()) {
        sources.push_back({iter_arg, delay, ii});
      }
    } else {
      for (mlir::Value result : item.operation->getResults()) {
        sources.push_back({result, delay, ii});
      }
    }

    // A yield is followed out to the loop result: one pass plus (trips - 1)
    // intervals, and the interval starts over.
    struct Consumer {
      mlir::Operation *operation;
      int32_t delay;
      int32_t ii;
    };
    llvm::SmallVector<Consumer> consumers;
    while (!sources.empty()) {
      Source s = sources.pop_back_val();
      for (mlir::OpOperand &use : s.value.getUses()) {
        mlir::Operation *owner = use.getOwner();
        if (mlir::isa<mlir::affine::AffineYieldOp>(owner)) {
          auto loop =
              mlir::dyn_cast<mlir::affine::AffineForOp>(owner->getParentOp());
          if (loop && use.getOperandNumber() < loop.getNumResults()) {
            int32_t trips = static_cast<int32_t>(trip_count(loop));
            int32_t scaled = trips > 0 ? s.delay + (trips - 1) * s.ii : 0;
            sources.push_back({loop.getResult(use.getOperandNumber()), scaled, 0});
          }
          continue;
        }
        consumers.push_back({owner, s.delay, s.ii});
      }
    }

    for (auto [consumer, consumer_delay, consumer_ii] : consumers) {
      auto rop = mlir::dyn_cast<vesyla::drra::RopOp>(consumer);
      if (rop && rop.getEvtAttr()) {
        reached.push_back({consumer, consumer_delay});
      } else {
        working_set.push_back({consumer_delay, consumer_ii, consumer});
      }
    }
  }
  return reached;
}

llvm::SmallVector<ReachedEvent> feeding_events(mlir::Operation *sink) {
  llvm::SmallVector<ReachedEvent> feeders;
  llvm::SmallVector<WorkItem> working_set{{0, 0, sink}};
  while (!working_set.empty()) {
    WorkItem item = working_set.pop_back_val();
    for (mlir::Value v : item.operation->getOperands()) {
      auto p = v.getDefiningOp<vesyla::drra::RopOp>();
      if (!p) {
        continue;
      }
      if (p.getEvtAttr()) {
        feeders.push_back({p, item.delay});
      } else {
        working_set.push_back({item.delay + rop_latency(p), 0, p});
      }
    }
  }
  return feeders;
}

llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>
tied_groups(mlir::Operation *scope) {
  // t(to) = t(from) + delay, kept in both directions.
  llvm::MapVector<mlir::Operation *,
                  llvm::SmallVector<std::pair<mlir::Operation *, int64_t>>>
      edges;
  scope->walk([&](vesyla::drra::RopOp rop) {
    if (!rop.getEvtAttr()) {
      return;
    }
    for (auto [event, delay] : reached_events(rop)) {
      edges[rop].push_back({event, delay});
      edges[event].push_back({rop, -static_cast<int64_t>(delay)});
    }
  });

  llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> groups;
  llvm::DenseMap<mlir::Operation *, int64_t> time;
  for (auto &entry : edges) {
    if (time.count(entry.first)) {
      continue;
    }
    llvm::SmallVector<mlir::Operation *> component{entry.first};
    time[entry.first] = 0;
    for (size_t i = 0; i < component.size(); ++i) {
      int64_t base = time[component[i]];
      for (auto [next, delay] : edges[component[i]]) {
        if (time.try_emplace(next, base + delay).second) {
          component.push_back(next);
        }
      }
    }
    llvm::MapVector<int64_t, llvm::SmallVector<mlir::Operation *>> at;
    for (mlir::Operation *op : component) {
      at[time[op]].push_back(op);
    }
    for (auto &same : at) {
      if (same.second.size() > 1) {
        groups.push_back(same.second);
      }
    }
  }
  return groups;
}

} // namespace vesyla::conversion::drra_to_pasm
