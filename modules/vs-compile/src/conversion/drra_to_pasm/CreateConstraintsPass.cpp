#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

#include "CreateConstraintsPass.hpp"

namespace vesyla::conversion::drra_to_pasm {
#define GEN_PASS_DEF_CREATECONSTRAINTSPASS
#include "conversion/drra_to_pasm/Passes.hpp.inc"

namespace {

// Marker set on a drra.rop once it has been processed, so the greedy driver
// converges (the pattern only matches rops without it) and stripped again once
// the driver finishes.
constexpr llvm::StringLiteral kProcessedMarker = "__cstr_processed__";

// One entry of the working set: the delay accumulated so far along the dataflow
// chain together with the operation to continue from. The operation may be a
// drra.rop or an affine.for whose iter_arg carries the value into the loop.
struct WorkItem {
  int32_t current_delay;
  mlir::Operation *operation;
};

// Enclosing affine.for loops of `op`, outermost first.
llvm::SmallVector<mlir::affine::AffineForOp>
get_enclosing_loops(mlir::Operation *op) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      loops.push_back(loop);
    }
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

// Build an anchor range from loop-derived indices. Loop dimensions map to IR
// (inner repetition) indices; there is no outer repetition or transition, so
// OR is empty and MT (the event id) is 0.
pasm::AnchorRangeAttr make_ir_anchor(mlir::MLIRContext *ctx,
                                     mlir::FlatSymbolRefAttr id,
                                     llvm::ArrayRef<uint32_t> ir_lo,
                                     llvm::ArrayRef<uint32_t> ir_hi) {
  return pasm::AnchorRangeAttr::get(ctx, id, /*or_lo=*/{}, /*mt_lo=*/0, ir_lo,
                                    /*or_hi=*/{}, /*mt_hi=*/0, ir_hi);
}

// Build an anchor for the given instruction symbol over `loops`: MT event 0 and
// one IR index dimension per loop, spanning [lower_bound, upper_bound - 1] (the
// affine.for upper bound is exclusive). With no loops the anchor is bare.
pasm::AnchorRangeAttr build_anchor(mlir::MLIRContext *ctx, mlir::FlatSymbolRefAttr id,
                             llvm::ArrayRef<mlir::affine::AffineForOp> loops) {
  if (loops.empty()) {
    return make_ir_anchor(ctx, id, {}, {});
  }

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
  return make_ir_anchor(ctx, id, idx_lo, idx_hi);
}

// Build a single-point anchor fixed to the last iteration of each enclosing
// loop (each dimension = upper_bound - 1). Bare when there are no loops.
pasm::AnchorRangeAttr build_last_anchor(mlir::MLIRContext *ctx, mlir::Operation *op,
                                  mlir::FlatSymbolRefAttr id) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops = get_enclosing_loops(op);
  if (loops.empty()) {
    return make_ir_anchor(ctx, id, {}, {});
  }

  llvm::SmallVector<uint32_t> idx;
  for (mlir::affine::AffineForOp loop : loops) {
    int64_t ub =
        loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
    idx.push_back(static_cast<uint32_t>(ub - 1));
  }
  return make_ir_anchor(ctx, id, idx, idx);
}

// Build a single-point anchor fixed to the first iteration of each enclosing
// loop (each dimension = lower_bound). Bare when there are no loops.
pasm::AnchorRangeAttr build_first_anchor(mlir::MLIRContext *ctx,
                                         mlir::Operation *op,
                                         mlir::FlatSymbolRefAttr id) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops = get_enclosing_loops(op);
  if (loops.empty()) {
    return make_ir_anchor(ctx, id, {}, {});
  }

  llvm::SmallVector<uint32_t> idx;
  for (mlir::affine::AffineForOp loop : loops) {
    int64_t lb =
        loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
    idx.push_back(static_cast<uint32_t>(lb));
  }
  return make_ir_anchor(ctx, id, idx, idx);
}

// The loop an operation's result initialises, when that loop then makes nothing
// of the value.
//
// A resource that accumulates holds the running total itself, so the operation
// that clears it produces the value the program carries into the loop as the
// iter_args initialiser, while the operation that accumulates takes no operand
// for it -- the register is the resource's own state and not an instruction
// operand. That leaves the region argument with no users, so the dataflow walk
// has nothing to follow out of it and the clear ends up constrained against
// nothing at all: the scheduler is then free to run every one of them back to
// back, ahead of the accumulations they are meant to punctuate.
//
// Returns the loop when the rop has that shape, and null otherwise -- a region
// argument that is used is an ordinary loop-carried value, which the dataflow
// walk handles on its own.
//
// TODO: this recognises an accumulator by the shape it leaves in the IR rather
// than by anything the resource says, which is the weak part. Nothing declares
// that one operation clears state another accumulates into; what is matched is
// that a value initialises a loop and the loop never reads it, and an unrelated
// operation that happens to produce a dead loop-carried value would be timed as
// if it were a clear. Two better answers were passed over for this one, both
// costlier: bind the accumulator through instruction selection so the
// dependency is an operand and every pass sees it, which needs the matcher to
// bind the pattern's rank-0 memref to the program's loop-carried value and the
// resulting operand to be kept out of `endpoints` and icdep generation; or let
// the library name the state, so a dpu rst says it writes `accumulate:0` and a
// mac says it reads it, the way `uses` already names the parts of a resource.
// The second is the smaller of the two and fits what is already there.
mlir::affine::AffineForOp accumulator_loop(vesyla::drra::RopOp rop) {
  for (mlir::Value result : rop->getResults()) {
    for (mlir::OpOperand &use : result.getUses()) {
      auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(use.getOwner());
      if (!loop) {
        continue;
      }
      unsigned control = loop.getNumControlOperands();
      if (use.getOperandNumber() < control) {
        continue;
      }
      unsigned index = use.getOperandNumber() - control;
      if (index >= loop.getRegionIterArgs().size()) {
        continue;
      }
      if (loop.getRegionIterArgs()[index].use_empty()) {
        return loop;
      }
    }
  }
  return nullptr;
}

// An anchor over `op`'s enclosing loops in which the loops it shares with the
// initialiser span their whole range and the loops inside those are pinned to
// their first iteration. That names one point per pass through the shared
// loops: the first time `op` runs within it, which is where the accumulation it
// belongs to begins.
pasm::AnchorRangeAttr
build_first_within(mlir::MLIRContext *ctx, mlir::Operation *op,
                   llvm::ArrayRef<mlir::affine::AffineForOp> shared,
                   mlir::FlatSymbolRefAttr id) {
  llvm::SmallVector<uint32_t> idx_lo;
  llvm::SmallVector<uint32_t> idx_hi;
  for (mlir::affine::AffineForOp loop : get_enclosing_loops(op)) {
    int64_t lb =
        loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
    int64_t ub =
        loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
    idx_lo.push_back(static_cast<uint32_t>(lb));
    idx_hi.push_back(
        static_cast<uint32_t>(llvm::is_contained(shared, loop) ? ub - 1 : lb));
  }
  return make_ir_anchor(ctx, id, idx_lo, idx_hi);
}

// Collect the (row, col, slot) of every resource an op touches (port ignored).
// The op's `resource` attribute is either a single pasm::ResourceAttr or an array.
void collect_resource_keys(
    mlir::Operation *op,
    std::set<std::tuple<int32_t, int32_t, int32_t>> &keys) {
  mlir::Attribute attr = op->getAttr("resource");
  if (auto res = mlir::dyn_cast_or_null<pasm::ResourceAttr>(attr)) {
    keys.insert({res.getRow(), res.getCol(), res.getSlot()});
  } else if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute e : arr) {
      if (auto res = mlir::dyn_cast<pasm::ResourceAttr>(e)) {
        keys.insert({res.getRow(), res.getCol(), res.getSlot()});
      }
    }
  }
}

// The symbol the conf of `op` lowers to. DrraToPasmPass splits a rop carrying
// both a conf and an evt into two pasm.rops -- the conf suffixed, the evt
// keeping the plain id, since they are placed differently and cannot share a
// symbol -- so a constraint anchored at the conf has to name it the same way.
mlir::FlatSymbolRefAttr conf_symbol(mlir::Operation *op,
                                    mlir::FlatSymbolRefAttr id) {
  auto rop = mlir::dyn_cast<vesyla::drra::RopOp>(op);
  if (rop && rop.getConfAttr() && rop.getEvtAttr()) {
    return mlir::FlatSymbolRefAttr::get(op->getContext(),
                                        id.getValue().str() + "_conf");
  }
  return id;
}

// Order a pair of symbol names so it can be looked up regardless of order.
std::pair<std::string, std::string> ordered_id_pair(llvm::StringRef a,
                                                    llvm::StringRef b) {
  return a < b ? std::make_pair(a.str(), b.str())
               : std::make_pair(b.str(), a.str());
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

// Time a clear against the loop it initialises: it has to happen as each pass
// through that loop begins, which is where the accumulation it clears starts.
//
// TODO: the timing here is fixed rather than derived. [0, 0] says the clear
// lands exactly on the first event of the pass, which is what this resource
// wants and what the hand-written pasm for it says, but a resource whose clear
// takes a cycle to land would need it stated -- another thing the library is
// the right place for.
//
// One constraint per event inside the loop rather than a chosen one of them.
// They all begin the pass together -- the reads that feed the accumulation are
// simultaneous -- so constraining the clear against each says the same thing
// twice rather than two different things, and it does not depend on which of
// them happens to be walked first. The operation that does the accumulating is
// not among them: a resource configured rather than triggered carries only a
// conf, which is lifted out of the loop entirely and has no per-pass event to
// anchor at.
void constrain_accumulator_clear(vesyla::drra::RopOp rop,
                                 mlir::affine::AffineForOp loop,
                                 mlir::PatternRewriter &rewriter) {
  auto rop_id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  pasm::EpochOp epoch = rop->getParentOfType<pasm::EpochOp>();
  if (!rop_id || !epoch) {
    return;
  }

  mlir::MLIRContext *ctx = rewriter.getContext();
  llvm::SmallVector<mlir::affine::AffineForOp> shared = get_enclosing_loops(rop);
  pasm::AnchorRangeAttr src = build_anchor(ctx, rop_id, shared);

  llvm::SmallVector<vesyla::drra::RopOp> events;
  loop.walk([&](vesyla::drra::RopOp inner) {
    if (inner.getEvtAttr()) {
      events.push_back(inner);
    }
  });

  rewriter.setInsertionPoint(epoch.getBody().front().getTerminator());
  for (vesyla::drra::RopOp event : events) {
    auto id = event->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
    if (!id) {
      continue;
    }
    pasm::AnchorRangeAttr dst = build_first_within(ctx, event, shared, id);
    pasm::CstrOp::create(rewriter, rop.getLoc(), src, dst,
                         pasm::DelayAttr::get(ctx, 0, 0),
                         rewriter.getBoolAttr(false));
  }
}

// Greedy rewrite pattern that triggers on every drra.rop. Each rop is processed
// exactly once: it is the starting point from which the dataflow chain is
// walked to create the delay constraints.
class CreateConstraintsRewriter
    : public mlir::OpRewritePattern<vesyla::drra::RopOp> {
public:
  using mlir::OpRewritePattern<vesyla::drra::RopOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(vesyla::drra::RopOp rop,
                  mlir::PatternRewriter &rewriter) const final {
    if (rop->hasAttr(kProcessedMarker)) {
      return mlir::failure();
    }
    // Only evts source a constraint; confs and other ops are walked through but
    // never originate one. A rop carrying both is an evt for this purpose --
    // its conf is hoisted away and constrained separately, below.
    if (!rop.getEvtAttr()) {
      return mlir::failure();
    }

    // Working set of items still to be processed, seeded with this rop at delay
    // 0.
    llvm::SmallVector<WorkItem> working_set;
    working_set.push_back({0, rop.getOperation()});

    while (!working_set.empty()) {
      WorkItem item = working_set.pop_back_val();

      // Retrieve the operation's own delay (missing attribute defaults to 0)
      // and add it to the delay accumulated so far.
      int32_t delay = item.current_delay;
      if (auto delay_attr =
              item.operation->getAttrOfType<mlir::IntegerAttr>("delay")) {
        delay += static_cast<int32_t>(delay_attr.getInt());
      }

      // Determine the source values to walk from, each tagged with the delay
      // carried to it. If the operation is an affine.for, it was entered through
      // an iter_args init, so walk the region iter arguments into the loop body.
      // Otherwise walk the operation's results.
      llvm::SmallVector<std::pair<mlir::Value, int32_t>> sources;
      if (auto loop =
              mlir::dyn_cast<mlir::affine::AffineForOp>(item.operation)) {
        for (mlir::BlockArgument iter_arg : loop.getRegionIterArgs()) {
          sources.push_back({iter_arg, delay});
        }
      } else {
        for (mlir::Value result : item.operation->getResults()) {
          sources.push_back({result, delay});
        }
      }

      // Collect the consumer ops, each tagged with the delay carried to it. A use
      // by an affine.yield is followed out to the matching loop result
      // (yield-out), multiplying the delay by that loop's trip count since the
      // loop body runs that many times.
      llvm::SmallVector<std::pair<mlir::Operation *, int32_t>> consumers;
      while (!sources.empty()) {
        auto [value, value_delay] = sources.pop_back_val();
        for (mlir::OpOperand &use : value.getUses()) {
          mlir::Operation *owner = use.getOwner();
          if (mlir::isa<mlir::affine::AffineYieldOp>(owner)) {
            auto loop =
                mlir::dyn_cast<mlir::affine::AffineForOp>(owner->getParentOp());
            if (loop && use.getOperandNumber() < loop.getNumResults()) {
              int32_t scaled =
                  value_delay * static_cast<int32_t>(trip_count(loop));
              sources.push_back(
                  {loop.getResult(use.getOperandNumber()), scaled});
            }
            continue;
          }
          consumers.push_back({owner, value_delay});
        }
      }

      // Visit every collected consumer.
      for (auto [consumer, consumer_delay] : consumers) {
        auto consumer_rop = mlir::dyn_cast<vesyla::drra::RopOp>(consumer);
        if (consumer_rop && consumer_rop.getEvtAttr()) {
          // Reached an evt: emit a pasm.cstr carrying the delay walked to it.
          mlir::MLIRContext *ctx = rewriter.getContext();
          llvm::SmallVector<mlir::affine::AffineForOp> rop_loops =
              get_enclosing_loops(rop);
          llvm::SmallVector<mlir::affine::AffineForOp> consumer_loops =
              get_enclosing_loops(consumer);
          // The src anchor carries every enclosing loop of the producer: loops
          // shared with the consumer span their full range [lb, ub-1], while
          // producer-only loops (the ones the dataflow leaves) are pinned to
          // their first iteration [lb, lb] — the remaining iterations are
          // already folded into the delay.
          pasm::AnchorRangeAttr src;
          auto rop_id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
          if (rop_loops.empty()) {
            src = make_ir_anchor(ctx, rop_id, {}, {});
          } else {
            llvm::SmallVector<uint32_t> src_lo;
            llvm::SmallVector<uint32_t> src_hi;
            for (mlir::affine::AffineForOp loop : rop_loops) {
              bool shared = false;
              for (mlir::affine::AffineForOp c : consumer_loops) {
                if (c.getOperation() == loop.getOperation()) {
                  shared = true;
                  break;
                }
              }
              int64_t lb =
                  loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
              int64_t ub =
                  loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
              src_lo.push_back(static_cast<uint32_t>(lb));
              src_hi.push_back(static_cast<uint32_t>(shared ? ub - 1 : lb));
            }
            src = make_ir_anchor(ctx, rop_id, src_lo, src_hi);
          }
          auto dst = build_anchor(
              ctx, consumer->getAttrOfType<mlir::FlatSymbolRefAttr>("id"),
              consumer_loops);
          auto delay_attr = pasm::DelayAttr::get(ctx, consumer_delay, consumer_delay);
          // Insert the constraint at the epoch level, before its terminator, so
          // it does not end up inside an affine.for body.
          pasm::EpochOp epoch = rop->getParentOfType<pasm::EpochOp>();
          rewriter.setInsertionPoint(epoch.getBody().front().getTerminator());
          pasm::CstrOp::create(rewriter, rop.getLoc(), src, dst, delay_attr,
                         rewriter.getBoolAttr(false));
        } else {
          // Not an evt drra.rop: keep walking by enqueuing the consumer with the
          // delay walked to it.
          working_set.push_back({consumer_delay, consumer});
        }
      }
    }

    // A clear whose loop makes nothing of the value it carries in leaves the
    // walk above with no consumer to reach, so it is timed against that loop
    // instead of against a chain that is not there.
    if (mlir::affine::AffineForOp loop = accumulator_loop(rop)) {
      constrain_accumulator_clear(rop, loop, rewriter);
    }

    rewriter.modifyOpInPlace(
        rop, [&] { rop->setAttr(kProcessedMarker, rewriter.getUnitAttr()); });
    return mlir::success();
  }
};

class CreateConstraintsPass
    : public impl::CreateConstraintsPassBase<CreateConstraintsPass> {
public:
  using impl::CreateConstraintsPassBase<
      CreateConstraintsPass>::CreateConstraintsPassBase;

  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<CreateConstraintsRewriter>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(),
            mlir::FrozenRewritePatternSet(std::move(patterns))))) {
      signalPassFailure();
      return;
    }

    // Strip the per-rop markers so later passes see clean drra.rop ops.
    getOperation().walk(
        [](vesyla::drra::RopOp rop) { rop->removeAttr(kProcessedMarker); });

    // Resource-contention constraints: two statements touching the same
    // (row, col, slot) resource (port ignored) must be at least one cycle apart
    // (delay [1, ]), unless a dataflow constraint already orders them. The
    // constraint runs from the first statement's last loop iteration to the
    // second statement's first loop iteration.
    mlir::MLIRContext *ctx = &getContext();
    getOperation().walk([&](pasm::EpochOp epoch) {
      mlir::OpBuilder builder(ctx);
      builder.setInsertionPoint(epoch.getBody().front().getTerminator());

      // Id pairs already ordered by a dataflow constraint (the cstr ops created
      // above), so they are not also given a resource-contention constraint.
      std::set<std::pair<std::string, std::string>> dataflow_pairs;
      epoch.walk([&](pasm::CstrOp cstr) {
        dataflow_pairs.insert(ordered_id_pair(cstr.getSrc().getInstr().getValue(),
                                              cstr.getDst().getInstr().getValue()));
      });

      llvm::SmallVector<mlir::Operation *> statements;
      llvm::SmallVector<mlir::Operation *> configs;
      epoch.walk([&](mlir::Operation *op) {
        if (!op->hasAttr("id") || !op->hasAttr("resource")) {
          return;
        }
        // A rop is a config where it carries a conf and a statement where it
        // carries an evt, so one carrying both is in both lists: its conf has
        // to be applied before its own evt fires, which is exactly the
        // first-use constraint below. Anything carrying neither is read as a
        // statement, which is what everything without a conf was before.
        auto rop = mlir::dyn_cast<vesyla::drra::RopOp>(op);
        if (rop && (rop.getConfAttr() || rop.getEvtAttr())) {
          if (rop.getConfAttr()) {
            configs.push_back(op);
          }
          if (rop.getEvtAttr()) {
            statements.push_back(op);
          }
          return;
        }
        statements.push_back(op);
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
          auto b_id =
              mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(b->getAttr("id"));
          if (!b_id) {
            continue;
          }
          if (dataflow_pairs.count(
                  ordered_id_pair(a_id.getValue(), b_id.getValue()))) {
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
          pasm::AnchorRangeAttr src = build_last_anchor(ctx, a, a_id);
          pasm::AnchorRangeAttr dst = build_first_anchor(ctx, b, b_id);
          auto delay_attr = pasm::DelayAttr::get(ctx, 1, std::nullopt);
          pasm::CstrOp::create(builder, a->getLoc(), src, dst, delay_attr,
                         builder.getBoolAttr(false));
        }
      }

      // Config first-use constraints: a config must be applied before the
      // resource it configures is first used. A config can set up several
      // resources at once, so each of its resource keys gets its own first use
      // — the first later statement (in program order) touching that key. Each
      // distinct use gets a delay [1, ] constraint. The config sits outside any
      // loop, so its src anchor is bare (no event, no indices); the use's dst
      // anchor still carries its enclosing loops.
      for (mlir::Operation *cfg : configs) {
        auto cfg_id =
            mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(cfg->getAttr("id"));
        if (!cfg_id) {
          continue;
        }
        std::set<std::tuple<int32_t, int32_t, int32_t>> cfg_keys;
        collect_resource_keys(cfg, cfg_keys);

        // Earliest use of each configured key, deduplicated but kept in the
        // order the keys are visited so the emitted constraints are stable.
        llvm::SmallVector<mlir::Operation *> first_uses;
        std::set<mlir::Operation *> seen;
        for (const auto &key : cfg_keys) {
          for (mlir::Operation *use : statements) {
            std::set<std::tuple<int32_t, int32_t, int32_t>> use_keys;
            collect_resource_keys(use, use_keys);
            if (use_keys.count(key)) {
              if (seen.insert(use).second) {
                first_uses.push_back(use);
              }
              break;
            }
          }
        }

        for (mlir::Operation *use : first_uses) {
          auto use_id =
              mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(use->getAttr("id"));
          if (!use_id) {
            continue;
          }
          auto src = make_ir_anchor(ctx, conf_symbol(cfg, cfg_id), {}, {});
          pasm::AnchorRangeAttr dst = build_first_anchor(ctx, use, use_id);
          auto delay_attr = pasm::DelayAttr::get(ctx, 1, std::nullopt);
          pasm::CstrOp::create(builder, cfg->getLoc(), src, dst, delay_attr,
                         builder.getBoolAttr(false));
        }
      }
    });
  }
};

} // namespace
} // namespace vesyla::conversion::drra_to_pasm
