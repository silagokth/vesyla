#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

#include "vesyla/Dialect/Pasm/Transforms/CreateConstraintsPass.hpp"
#include "vesyla/Support/Config.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_CREATECONSTRAINTSPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Collect the affine.for loops enclosing `op` up to (but excluding) `epoch`,
// ordered outermost to innermost.
llvm::SmallVector<mlir::affine::AffineForOp>
get_enclosing_loops(mlir::Operation *op, EpochOp epoch) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops;
  for (mlir::Operation *parent = op->getParentOp();
       parent && parent != epoch.getOperation();
       parent = parent->getParentOp()) {
    if (auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      loops.push_back(loop);
    }
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

// Build an anchor range for `id` from its enclosing loops. With no loops the
// anchor is bare (empty event/indices). With loops, the event is "e0" and each
// dimension spans [lower_bound, upper_bound - 1]: the affine.for upper bound is
// exclusive, so the inclusive high index is one less.
AnchorRangeAttr
build_anchor_range(mlir::MLIRContext *ctx, mlir::FlatSymbolRefAttr id,
                   llvm::ArrayRef<mlir::affine::AffineForOp> loops) {
  llvm::SmallVector<uint32_t> idx_lo;
  llvm::SmallVector<uint32_t> idx_hi;
  for (mlir::affine::AffineForOp loop : loops) {
    int64_t lb =
        loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
    int64_t ub =
        loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
    idx_lo.push_back(static_cast<uint32_t>(lb));
    idx_hi.push_back(static_cast<uint32_t>(ub - 1));
  }
  llvm::StringRef event = loops.empty() ? "" : "e0";
  return AnchorRangeAttr::get(ctx, id, event, idx_lo, idx_hi);
}

// Number of iterations of an affine.for: (upper - lower) / step.
int64_t trip_count(mlir::affine::AffineForOp loop) {
  int64_t lb = loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
  int64_t ub = loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 0;
  int64_t step = loop.getStepAsInt();
  if (step == 0) {
    return 0;
  }
  return (ub - lb) / step;
}

// Collect the effective consumers of `value`: ops that use it directly, except
// that a use by an affine.yield is followed through the enclosing affine.for's
// matching result (a loop-carried dependency) until a non-yield use is reached.
void collect_effective_consumers(
    mlir::Value value, llvm::SmallVectorImpl<mlir::Operation *> &out) {
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *consumer = use.getOwner();
    if (mlir::isa<mlir::affine::AffineYieldOp>(consumer)) {
      auto loop =
          mlir::dyn_cast<mlir::affine::AffineForOp>(consumer->getParentOp());
      if (loop && use.getOperandNumber() < loop.getNumResults()) {
        collect_effective_consumers(loop.getResult(use.getOperandNumber()),
                                    out);
      }
      continue;
    }
    out.push_back(consumer);
  }
}

// Build a single-point anchor fixed to the last iteration of each enclosing
// loop (each dimension = upper_bound - 1, the exclusive affine.for upper bound
// minus one). Bare (empty event/indices) when there are no loops.
AnchorRangeAttr
build_last_anchor(mlir::MLIRContext *ctx, mlir::FlatSymbolRefAttr id,
                  llvm::ArrayRef<mlir::affine::AffineForOp> loops) {
  llvm::SmallVector<uint32_t> idx;
  for (mlir::affine::AffineForOp loop : loops) {
    int64_t ub =
        loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
    idx.push_back(static_cast<uint32_t>(ub - 1));
  }
  llvm::StringRef event = loops.empty() ? "" : "e0";
  return AnchorRangeAttr::get(ctx, id, event, idx, idx);
}

// Collect the (row, col, slot) of every resource an op touches (port ignored).
// The op's `resource` attribute is either a single ResourceAttr or an array.
void collect_resource_keys(
    mlir::Operation *op,
    std::set<std::tuple<int32_t, int32_t, int32_t>> &keys) {
  mlir::Attribute attr = op->getAttr("resource");
  if (auto res = mlir::dyn_cast_or_null<ResourceAttr>(attr)) {
    keys.insert({res.getRow(), res.getCol(), res.getSlot()});
  } else if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute e : arr) {
      if (auto res = mlir::dyn_cast<ResourceAttr>(e)) {
        keys.insert({res.getRow(), res.getCol(), res.getSlot()});
      }
    }
  }
}

// Order a pair of ops by pointer so it can be looked up regardless of which op
// is first.
std::pair<mlir::Operation *, mlir::Operation *>
ordered_pair(mlir::Operation *a, mlir::Operation *b) {
  return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}

// Resolve the output delay for an instruction from the ISA. Looks up the
// component by `kind` and the instruction by `name`, then reads its `delay`:
// an integer is used directly; a string names one of the instruction's
// segments, and the op's `key` attribute selects that segment's verbo_map
// entry, whose `delay` field is the value. Returns false if the delay cannot
// be resolved.
bool lookup_delay(const nlohmann::json &isa, mlir::Operation *op,
                  llvm::StringRef kind, llvm::StringRef name,
                  int32_t &out_delay) {
  if (!isa.is_object() || !isa.contains("components") ||
      !isa["components"].is_array()) {
    return false;
  }
  for (const auto &component : isa["components"]) {
    if (!component.is_object() || !component.contains("kind") ||
        !component["kind"].is_string() ||
        component["kind"].get<std::string>() != kind) {
      continue;
    }
    if (!component.contains("instructions") ||
        !component["instructions"].is_array()) {
      return false;
    }
    for (const auto &instr : component["instructions"]) {
      if (!instr.is_object() || !instr.contains("name") ||
          !instr["name"].is_string() ||
          instr["name"].get<std::string>() != name) {
        continue;
      }
      if (!instr.contains("delay")) {
        return false;
      }
      const auto &d = instr["delay"];
      // A numeric delay is used directly.
      if (d.is_number_integer()) {
        out_delay = d.get<int32_t>();
        return true;
      }
      // A string delay names a segment; the op's `key` attribute selects the
      // verbo_map entry of that segment, whose `delay` field is the value.
      if (d.is_string()) {
        std::string seg_name = d.get<std::string>();
        auto key_attr =
            mlir::dyn_cast_or_null<mlir::IntegerAttr>(op->getAttr("key"));
        if (!key_attr) {
          return false;
        }
        int64_t key_val = key_attr.getInt();
        if (!instr.contains("segments") || !instr["segments"].is_array()) {
          return false;
        }
        for (const auto &seg : instr["segments"]) {
          if (!seg.is_object() || !seg.contains("name") ||
              !seg["name"].is_string() ||
              seg["name"].get<std::string>() != seg_name) {
            continue;
          }
          if (!seg.contains("verbo_map") || !seg["verbo_map"].is_array()) {
            return false;
          }
          for (const auto &entry : seg["verbo_map"]) {
            if (entry.is_object() && entry.contains("key") &&
                entry["key"].is_number_integer() &&
                entry["key"].get<int64_t>() == key_val) {
              if (entry.contains("delay") &&
                  entry["delay"].is_number_integer()) {
                out_delay = entry["delay"].get<int32_t>();
                return true;
              }
              return false;
            }
          }
          return false;
        }
      }
      return false;
    }
    return false;
  }
  return false;
}

class CreateConstraintsPass
    : public impl::CreateConstraintsPassBase<CreateConstraintsPass> {
public:
  using impl::CreateConstraintsPassBase<
      CreateConstraintsPass>::CreateConstraintsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    Config cfg;
    nlohmann::json isa = cfg.get_isa_json();

    bool failed = false;
    module.walk([&](EpochOp epoch) {
      mlir::Block &block = epoch.getBody().front();
      mlir::OpBuilder builder(ctx);
      builder.setInsertionPoint(block.getTerminator());

      // Op pairs that get a dataflow constraint; used to suppress a redundant
      // resource-contention constraint between the same two statements.
      std::set<std::pair<mlir::Operation *, mlir::Operation *>> dataflow_pairs;

      // Gather every result-producing op that carries id/instr/kind.
      llvm::SmallVector<mlir::Operation *> producers;
      epoch.walk([&](mlir::Operation *op) {
        if (op->getNumResults() > 0 && op->hasAttr("id") &&
            op->hasAttr("instr") && op->hasAttr("kind")) {
          producers.push_back(op);
        }
      });

      for (mlir::Operation *producer : producers) {
        auto producer_id =
            mlir::dyn_cast<mlir::FlatSymbolRefAttr>(producer->getAttr("id"));
        auto instr_attr =
            mlir::dyn_cast<mlir::StringAttr>(producer->getAttr("instr"));
        auto kind_attr =
            mlir::dyn_cast<mlir::StringAttr>(producer->getAttr("kind"));
        if (!producer_id || !instr_attr || !kind_attr) {
          continue;
        }

        int32_t delay = 0;
        if (!lookup_delay(isa, producer, kind_attr.getValue(),
                          instr_attr.getValue(), delay)) {
          producer->emitError()
              << "CreateConstraintsPass: no output delay found in ISA for "
                 "kind '"
              << kind_attr.getValue() << "', instr '" << instr_attr.getValue()
              << "'";
          failed = true;
          continue;
        }

        llvm::SmallVector<mlir::affine::AffineForOp> producer_loops =
            get_enclosing_loops(producer, epoch);

        // One constraint per (producer result -> effective consumer) edge.
        // Effective consumers follow affine.yield through the enclosing loop's
        // result, so loop-carried dependencies reach their real consumer.
        for (mlir::OpResult result : producer->getResults()) {
          llvm::SmallVector<mlir::Operation *> consumers;
          collect_effective_consumers(result, consumers);
          for (mlir::Operation *consumer : consumers) {
            auto consumer_id = mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(
                consumer->getAttr("id"));
            if (!consumer_id) {
              continue;
            }
            llvm::SmallVector<mlir::affine::AffineForOp> consumer_loops =
                get_enclosing_loops(consumer, epoch);

            // The anchors carry only the loops shared by producer and consumer
            // (the outer loops). Loops enclosing the producer but not the
            // consumer are "collapsed" (the value was yielded out of them), and
            // the delay is multiplied by their trip counts.
            llvm::SmallVector<mlir::affine::AffineForOp> common;
            int64_t multiplier = 1;
            for (mlir::affine::AffineForOp loop : producer_loops) {
              bool shared = false;
              for (mlir::affine::AffineForOp c : consumer_loops) {
                if (c.getOperation() == loop.getOperation()) {
                  shared = true;
                  break;
                }
              }
              if (shared) {
                common.push_back(loop);
              } else {
                multiplier *= trip_count(loop);
              }
            }

            AnchorRangeAttr src = build_anchor_range(ctx, producer_id, common);
            AnchorRangeAttr dst =
                build_anchor_range(ctx, consumer_id, consumer_loops);
            int32_t edge_delay = static_cast<int32_t>(delay * multiplier);
            auto delay_attr = DelayAttr::get(ctx, edge_delay, edge_delay);
            CstrOp::create(builder, producer->getLoc(), src, dst, delay_attr,
                           builder.getBoolAttr(false));
            dataflow_pairs.insert(ordered_pair(producer, consumer));
          }
        }
      }

      // Resource-contention constraints: two statements that touch the same
      // (row, col, slot) resource (port ignored) must be at least one cycle
      // apart (delay [1, ]), unless a dataflow constraint already orders them.
      // The earlier statement (program order) is src, the later is dst, and
      // each anchor is fixed to its last loop iteration.
      llvm::SmallVector<mlir::Operation *> statements;
      epoch.walk([&](mlir::Operation *op) {
        if (op->hasAttr("id") && op->hasAttr("resource")) {
          statements.push_back(op);
        }
      });
      for (size_t i = 0; i < statements.size(); ++i) {
        mlir::Operation *a = statements[i];
        auto a_id =
            mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(a->getAttr("id"));
        if (!a_id) {
          continue;
        }
        std::set<std::tuple<int32_t, int32_t, int32_t>> a_keys;
        collect_resource_keys(a, a_keys);
        for (size_t j = i + 1; j < statements.size(); ++j) {
          mlir::Operation *b = statements[j];
          if (dataflow_pairs.count(ordered_pair(a, b))) {
            continue;
          }
          auto b_id =
              mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(b->getAttr("id"));
          if (!b_id) {
            continue;
          }
          std::set<std::tuple<int32_t, int32_t, int32_t>> b_keys;
          collect_resource_keys(b, b_keys);
          bool shares = false;
          for (const auto &k : a_keys) {
            if (b_keys.count(k)) {
              shares = true;
              break;
            }
          }
          if (!shares) {
            continue;
          }
          AnchorRangeAttr src =
              build_last_anchor(ctx, a_id, get_enclosing_loops(a, epoch));
          AnchorRangeAttr dst =
              build_last_anchor(ctx, b_id, get_enclosing_loops(b, epoch));
          auto delay_attr = DelayAttr::get(ctx, 1, std::nullopt);
          CstrOp::create(builder, a->getLoc(), src, dst, delay_attr,
                         builder.getBoolAttr(false));
        }
      }
    });

    if (failed) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
