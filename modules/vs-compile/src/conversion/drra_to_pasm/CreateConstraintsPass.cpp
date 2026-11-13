#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/OptionLimit.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

#include "CreateConstraintsPass.hpp"
#include "DataflowWalk.hpp"

namespace vesyla::conversion::drra_to_pasm {
#define GEN_PASS_DEF_CREATECONSTRAINTSPASS
#include "conversion/drra_to_pasm/Passes.hpp.inc"

namespace {

// Marker set on a drra.rop once it has been processed, so the greedy driver
// converges (the pattern only matches rops without it) and stripped again once
// the driver finishes.
constexpr llvm::StringLiteral kProcessedMarker = "__cstr_processed__";

// What the debug dumps show against a constraint: the pass that emitted it and
// the rule that fired.
mlir::ArrayAttr debug_attr(mlir::Builder &builder, llvm::StringRef rule) {
  return builder.getStrArrayAttr({"CreateConstraintsPass", rule});
}

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
struct Accumulator {
  mlir::affine::AffineForOp loop;
  unsigned index; // the dead iter_arg
};

std::optional<Accumulator> accumulator_loop(vesyla::drra::RopOp rop) {
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
        return Accumulator{loop, index};
      }
    }
  }
  return std::nullopt;
}

// Which iteration of a loop that is not shared an anchor names.
enum class Pin { first, last };

// An anchor over `op`'s enclosing loops in which the loops it shares with the
// other end span their whole range and the loops inside those are pinned to a
// single iteration. That names one point per pass through the shared loops:
// with Pin::first the first time `op` runs within the pass, with Pin::last the
// last.
//
// Two anchors built this way over the same `shared` hold the same number of
// points -- one per pass -- which is what CstrOp::verify requires and what lets
// the timing model step them in lockstep, pairing pass with pass. With no
// shared loops both collapse to a single point, so the pair degenerates to the
// plain last-to-first ordering.
//
// The range is kept whole rather than sampled down to the points a solver
// needs: pasm.cstr is read by the interconnect pass too, which walks the
// constraint graph and maps an index through a pair by where it falls in these
// rectangles. Sampling belongs in the timing model, which is the only consumer
// that pays per point.
pasm::AnchorRangeAttr
build_within(mlir::MLIRContext *ctx, mlir::Operation *op,
             llvm::ArrayRef<mlir::affine::AffineForOp> shared,
             mlir::FlatSymbolRefAttr id, Pin pin) {
  llvm::SmallVector<uint32_t> idx_lo;
  llvm::SmallVector<uint32_t> idx_hi;
  for (mlir::affine::AffineForOp loop : get_enclosing_loops(op)) {
    int64_t lb =
        loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
    int64_t ub =
        loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
    if (llvm::is_contained(shared, loop)) {
      idx_lo.push_back(static_cast<uint32_t>(lb));
      idx_hi.push_back(static_cast<uint32_t>(ub - 1));
    } else {
      uint32_t at = static_cast<uint32_t>(pin == Pin::last ? ub - 1 : lb);
      idx_lo.push_back(at);
      idx_hi.push_back(at);
    }
  }
  return make_ir_anchor(ctx, id, idx_lo, idx_hi);
}

// The enclosing loops two operations have in common, outermost first.
llvm::SmallVector<mlir::affine::AffineForOp>
shared_loops(mlir::Operation *a, mlir::Operation *b) {
  llvm::SmallVector<mlir::affine::AffineForOp> b_loops = get_enclosing_loops(b);
  llvm::SmallVector<mlir::affine::AffineForOp> shared;
  for (mlir::affine::AffineForOp loop : get_enclosing_loops(a)) {
    if (llvm::is_contained(b_loops, loop)) {
      shared.push_back(loop);
    }
  }
  return shared;
}

// The storage a statement lands in, or null when it touches none. Instruction
// selection puts this on every rop whose resource holds storage of its own, so
// two rops naming the same one are two accesses to the same array.
mlir::FlatSymbolRefAttr storage_of(mlir::Operation *op) {
  return op->getAttrOfType<mlir::FlatSymbolRefAttr>("storage");
}

// Minimum delay from `src` to `dst` on one storage. A write may land in the
// cycle of the last read of the old data (the read sees the old value), so a
// read followed by a write needs 0; everything else 1.
int64_t storage_delay(mlir::Operation *src, mlir::Operation *dst) {
  bool src_reads = src->getNumResults() > 0;
  bool dst_writes = dst->getNumResults() == 0 && dst->getNumOperands() > 0;
  return src_reads && dst_writes ? 0 : 1;
}

// Whether `op`'s address varies with the enclosing loop at `depth`, outermost
// first. Instruction selection lifts the access's affine map onto the rop with
// one dimension per enclosing loop in that order -- NativeHelpers.cpp builds it
// that way when the access has no affine.apply of its own, and a program that
// does have one lists its induction variables the same way.
//
// A loop the address does not vary with is one the storage is reused across:
// every pass writes the same addresses over again. Anything unstated -- no map,
// or an address that is not known until the program runs -- is read as not
// varying, so the storage is taken to be reused unless the map says otherwise.
bool address_varies_with(mlir::Operation *op, unsigned depth) {
  auto map_attr = op->getAttrOfType<mlir::AffineMapAttr>("map");
  if (!map_attr) {
    return false;
  }
  mlir::AffineMap map = map_attr.getAffineMap();
  if (depth >= map.getNumDims()) {
    return false;
  }
  return map.isFunctionOfDim(depth);
}

// For each shared loop, whether the pair may meet again on a later pass through
// it. Two accesses to one storage do unless both addresses move with the loop;
// a pair that only contends for a slot always does, the slot being the same one
// every pass. Shared loops are the outermost ones and so occupy the same
// positions in both operations' enclosing lists, which is what lets one depth
// index stand for both.
llvm::SmallVector<bool>
reused_loops(mlir::Operation *a, mlir::Operation *b,
             llvm::ArrayRef<mlir::affine::AffineForOp> shared,
             bool same_storage) {
  llvm::SmallVector<bool> reused;
  for (size_t depth = 0; depth < shared.size(); ++depth) {
    reused.push_back(!same_storage || !address_varies_with(a, depth) ||
                     !address_varies_with(b, depth));
  }
  return reused;
}

// An anchor like build_within, except that the shared loop at `carry` spans one
// pass less and starts `offset` passes in: [lb + offset, ub - 2 + offset]. Two
// of these at offset 0 and 1 hold the same number of points and so pair pass i
// with pass i + 1, which is how a dependence that runs from one pass into the
// next is stated.
//
// A reused loop inside `carry` is pinned rather than spanned: stepping the
// outer loop leaves the last pass of the inner one and enters its first, so
// the two ends are that last and that first, not the same pass of each.
pasm::AnchorRangeAttr
build_carry(mlir::MLIRContext *ctx, mlir::Operation *op,
            llvm::ArrayRef<mlir::affine::AffineForOp> shared,
            llvm::ArrayRef<bool> reused, mlir::FlatSymbolRefAttr id, Pin pin,
            size_t carry, int64_t offset,
            llvm::ArrayRef<int64_t> inner = {}) {
  llvm::SmallVector<uint32_t> idx_lo;
  llvm::SmallVector<uint32_t> idx_hi;
  size_t depth = 0;
  for (mlir::affine::AffineForOp loop : get_enclosing_loops(op)) {
    int64_t lb =
        loop.hasConstantLowerBound() ? loop.getConstantLowerBound() : 0;
    int64_t ub =
        loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
    if (!inner.empty() && depth >= shared.size()) {
      // An explicit point in the loops the pair does not share.
      uint32_t at = static_cast<uint32_t>(inner[depth - shared.size()]);
      idx_lo.push_back(at);
      idx_hi.push_back(at);
    } else if (depth == carry) {
      idx_lo.push_back(static_cast<uint32_t>(lb + offset));
      idx_hi.push_back(static_cast<uint32_t>(ub - 2 + offset));
    } else if (llvm::is_contained(shared, loop) &&
               !(depth > carry && reused[depth])) {
      idx_lo.push_back(static_cast<uint32_t>(lb));
      idx_hi.push_back(static_cast<uint32_t>(ub - 1));
    } else {
      uint32_t at = static_cast<uint32_t>(pin == Pin::last ? ub - 1 : lb);
      idx_lo.push_back(at);
      idx_hi.push_back(at);
    }
    ++depth;
  }
  return make_ir_anchor(ctx, id, idx_lo, idx_hi);
}

// Two io reads that share nothing but the external read port. They only
// must not use it in the same cycle; neither writes a storage the other uses.
bool share_only_external_read(mlir::Operation *a, mlir::Operation *b) {
  auto kind_a = a->getAttrOfType<mlir::StringAttr>("kind");
  auto kind_b = b->getAttrOfType<mlir::StringAttr>("kind");
  auto uses_a = a->getAttrOfType<mlir::ArrayAttr>("uses");
  auto uses_b = b->getAttrOfType<mlir::ArrayAttr>("uses");
  if (!kind_a || !kind_b || kind_a.getValue() != "io" ||
      kind_b.getValue() != "io" || !uses_a || !uses_b) {
    return false;
  }
  llvm::SmallVector<llvm::StringRef> shared;
  for (mlir::Attribute u : uses_a) {
    if (llvm::is_contained(uses_b, u)) {
      shared.push_back(mlir::cast<mlir::StringAttr>(u).getValue());
    }
  }
  return shared.size() == 1 && shared.front() == "external_read:0";
}

// The points of `op` in the loops it does not share, at most `limit` of
// them, or nothing.
std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
inner_points(mlir::Operation *op, size_t num_shared, size_t limit) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops = get_enclosing_loops(op);
  llvm::SmallVector<llvm::SmallVector<int64_t>> points{{}};
  for (size_t d = num_shared; d < loops.size(); ++d) {
    if (!loops[d].hasConstantBounds()) {
      return std::nullopt;
    }
    llvm::SmallVector<llvm::SmallVector<int64_t>> next;
    for (auto &p : points) {
      for (int64_t i = loops[d].getConstantLowerBound();
           i < loops[d].getConstantUpperBound(); ++i) {
        next.push_back(p);
        next.back().push_back(i);
      }
    }
    points = std::move(next);
    if (points.size() > limit) {
      return std::nullopt;
    }
  }
  return points;
}

// Words per row when `op` moves a whole row (its value is a vector), else 0.
int64_t bulk_words(mlir::Operation *op) {
  mlir::Value value = op->getNumResults() > 0 ? op->getResult(0)
                      : op->getNumOperands() > 0 ? op->getOperand(0)
                                                 : mlir::Value();
  auto vec = value ? mlir::dyn_cast<mlir::VectorType>(value.getType())
                   : mlir::VectorType();
  return vec ? vec.getNumElements() : 0;
}

// The row `op` touches at each point of the loops it does not share, in
// program order, or nothing when that is not known: no single-result map, an
// address that moves with a shared loop, or too many points.
std::optional<llvm::SmallVector<std::pair<llvm::SmallVector<int64_t>, int64_t>>>
rows_per_point(mlir::Operation *op, size_t num_shared, int64_t row_words) {
  auto map_attr = op->getAttrOfType<mlir::AffineMapAttr>("map");
  if (!map_attr || map_attr.getAffineMap().getNumResults() != 1) {
    return std::nullopt;
  }
  mlir::AffineMap map = map_attr.getAffineMap();
  llvm::SmallVector<mlir::affine::AffineForOp> loops = get_enclosing_loops(op);
  if (map.getNumDims() != loops.size() || map.getNumSymbols() != 0) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t> lb, ub;
  int64_t points = 1;
  for (size_t d = 0; d < loops.size(); ++d) {
    if (!loops[d].hasConstantLowerBound() ||
        !loops[d].hasConstantUpperBound() || loops[d].getStepAsInt() != 1) {
      return std::nullopt;
    }
    if (d < num_shared && map.isFunctionOfDim(d)) {
      return std::nullopt;
    }
    lb.push_back(loops[d].getConstantLowerBound());
    ub.push_back(loops[d].getConstantUpperBound());
    if (d >= num_shared) {
      points *= ub.back() - lb.back();
    }
  }
  if (points <= 0 || points > 4096) {
    return std::nullopt;
  }
  int64_t words = bulk_words(op) ? 1 : row_words;
  llvm::SmallVector<std::pair<llvm::SmallVector<int64_t>, int64_t>> out;
  llvm::SmallVector<int64_t> at(lb);
  while (true) {
    int64_t addr = map.compose(at).front();
    out.push_back({llvm::SmallVector<int64_t>(at.begin() + num_shared,
                                              at.end()),
                   addr / words});
    size_t d = loops.size();
    while (d > num_shared && ++at[d - 1] == ub[d - 1]) {
      at[d - 1] = lb[d - 1];
      --d;
    }
    if (d == num_shared) {
      break;
    }
  }
  return out;
}

// Per RF row, the last point at which `reader` reads it and the first at which
// `writer` writes it, so a write waits only for the reads of its own row.
// Nothing when the rows are not known or the two share no row.
llvm::SmallVector<std::pair<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>>>
row_war_points(mlir::Operation *reader, mlir::Operation *writer,
               size_t num_shared) {
  int64_t row_words = std::max(bulk_words(reader), bulk_words(writer));
  if (row_words == 0) {
    return {};
  }
  auto reads = rows_per_point(reader, num_shared, row_words);
  auto writes = rows_per_point(writer, num_shared, row_words);
  if (!reads || !writes) {
    return {};
  }
  std::map<int64_t, llvm::SmallVector<int64_t>> last_read;
  for (auto &[point, row] : *reads) {
    last_read[row] = point;
  }
  std::map<int64_t, llvm::SmallVector<int64_t>> first_write;
  for (auto &[point, row] : *writes) {
    first_write.try_emplace(row, point);
  }
  llvm::SmallVector<std::pair<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>>> out;
  for (auto &[row, point] : first_write) {
    auto it = last_read.find(row);
    if (it != last_read.end()) {
      out.push_back({it->second, point});
    }
  }
  return out;
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

// Stream steps: an accumulating mode takes an operand every `ii` cycles, so
// every word event in a carried loop that holds a mode steps at exactly that
// ii. The dataflow delays assume it, being anchored on the first iteration
// only, while the contention rules may stretch the step: two streams on one
// port then get an interleaved schedule that satisfies everything, and the
// mode accumulates garbage in the gaps. A loop without a carry is left free:
// its modes are stateless, and a stretched step only costs cycles.
void constrain_stream_steps(pasm::EpochOp epoch, mlir::OpBuilder &builder) {
  mlir::MLIRContext *ctx = builder.getContext();
  epoch.walk([&](mlir::affine::AffineForOp loop) {
    if (trip_count(loop) < 2 || loop.getNumIterOperands() == 0) {
      return;
    }
    int32_t ii = 0;
    for (mlir::Operation &op : loop.getBody()->without_terminator()) {
      if (op.hasAttr("conf") && op.hasAttr("ii")) {
        ii = std::max(ii, static_cast<int32_t>(
                              op.getAttrOfType<mlir::IntegerAttr>("ii").getInt()));
      }
    }
    if (ii == 0) {
      return;
    }
    for (mlir::Operation &op : loop.getBody()->without_terminator()) {
      auto id = op.getAttrOfType<mlir::FlatSymbolRefAttr>("id");
      if (!id || !op.hasAttr("evt") || bulk_words(&op) != 0) {
        continue;
      }
      llvm::SmallVector<mlir::affine::AffineForOp> loops =
          get_enclosing_loops(&op);
      llvm::SmallVector<uint32_t> lo, hi;
      for (mlir::affine::AffineForOp enclosing : loops) {
        lo.push_back(static_cast<uint32_t>(enclosing.getConstantLowerBound()));
        hi.push_back(static_cast<uint32_t>(enclosing.getConstantUpperBound() - 1));
      }
      llvm::SmallVector<uint32_t> src_hi(hi), dst_lo(lo);
      src_hi.back() -= 1;
      dst_lo.back() += 1;
      pasm::CstrOp::create(builder, op.getLoc(),
                           make_ir_anchor(ctx, id, lo, src_hi),
                           make_ir_anchor(ctx, id, dst_lo, hi),
                           pasm::DelayAttr::get(ctx, ii, ii),
                           builder.getBoolAttr(false))
          ->setAttr("debug", debug_attr(builder, "stream-step"));
    }
  });
}

// Time a clear against the loop it initialises: it has to happen as each pass
// through that loop begins, which is where the accumulation it clears starts.
//
// TODO: the timing here is fixed rather than derived. Delay 0 from a direct
// feeder says the clear lands exactly on the first event of the pass, which is
// what this resource wants and what the hand-written pasm for it says, but a
// resource whose clear takes a cycle to land would need it stated -- another
// thing the library is the right place for.
//
// One constraint per evt that feeds the accumulating op (the one yielded into
// the dead iter_arg), with that evt's latency to it: a direct operand starts
// the pass with the clear, one that goes through another DPU first
// (acc += a * (b * c)) starts that DPU's latency earlier. Constraining every
// feeder says the same thing as the exact dataflow delays, so it does not
// depend on which is walked first. The accumulating op itself has no evt: it
// carries only a conf, lifted out of the loop.
//
// A loop whose yield is not a rop, or whose accumulator has no feeder inside
// the loop, falls back to every evt in the loop at delay 0.
void constrain_accumulator_clear(vesyla::drra::RopOp rop,
                                 Accumulator acc,
                                 mlir::PatternRewriter &rewriter) {
  auto rop_id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
  pasm::EpochOp epoch = rop->getParentOfType<pasm::EpochOp>();
  if (!rop_id || !epoch) {
    return;
  }

  mlir::MLIRContext *ctx = rewriter.getContext();
  llvm::SmallVector<mlir::affine::AffineForOp> shared = get_enclosing_loops(rop);
  pasm::AnchorRangeAttr clear = build_anchor(ctx, rop_id, shared);

  llvm::SmallVector<ReachedEvent> events;
  mlir::Value yielded =
      acc.loop.getBody()->getTerminator()->getOperand(acc.index);
  if (auto sink = yielded.getDefiningOp<vesyla::drra::RopOp>()) {
    for (const ReachedEvent &feeder : feeding_events(sink)) {
      if (acc.loop->isAncestor(feeder.event)) {
        events.push_back(feeder);
      }
    }
  }
  if (events.empty()) {
    acc.loop.walk([&](vesyla::drra::RopOp inner) {
      if (inner.getEvtAttr()) {
        events.push_back({inner, 0});
      }
    });
  }

  rewriter.setInsertionPoint(epoch.getBody().front().getTerminator());
  for (const ReachedEvent &event : events) {
    auto id = event.event->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
    if (!id) {
      continue;
    }
    pasm::AnchorRangeAttr first =
        build_within(ctx, event.event, shared, id, Pin::first);
    pasm::CstrOp::create(rewriter, rop.getLoc(), first, clear,
                         pasm::DelayAttr::get(ctx, event.delay, event.delay),
                         rewriter.getBoolAttr(false))
        ->setAttr("debug", debug_attr(rewriter, "accumulator-clear"));
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

    for (auto [consumer, consumer_delay] : reached_events(rop)) {
      // Reached an evt: emit a pasm.cstr carrying the delay walked to it.
      mlir::MLIRContext *ctx = rewriter.getContext();
      llvm::SmallVector<mlir::affine::AffineForOp> consumer_loops =
          get_enclosing_loops(consumer);
      llvm::SmallVector<mlir::affine::AffineForOp> shared =
          shared_loops(rop, consumer);
      auto rop_id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
      auto consumer_id = consumer->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
      auto delay_attr = pasm::DelayAttr::get(ctx, consumer_delay, consumer_delay);
      // Insert the constraints at the epoch level, before its terminator, so
      // they do not end up inside an affine.for body.
      pasm::EpochOp epoch = rop->getParentOfType<pasm::EpochOp>();
      rewriter.setInsertionPoint(epoch.getBody().front().getTerminator());
      // Loops shared with the consumer span their full range; the ones the
      // dataflow leaves are pinned to their first iteration, their remaining
      // iterations being already folded into the delay.
      pasm::AnchorRangeAttr src =
          build_within(ctx, rop, shared, rop_id, Pin::first);
      pasm::AnchorRangeAttr dst = build_anchor(ctx, consumer_id, consumer_loops);
      pasm::CstrOp::create(rewriter, rop.getLoc(), src, dst, delay_attr,
                           rewriter.getBoolAttr(false))
          ->setAttr("debug", debug_attr(rewriter, "dataflow"));
    }

    // A clear whose loop makes nothing of the value it carries in leaves the
    // walk above with no consumer to reach, so it is timed against that loop
    // instead of against a chain that is not there.
    if (std::optional<Accumulator> acc = accumulator_loop(rop)) {
      constrain_accumulator_clear(rop, *acc, rewriter);
    }

    rewriter.modifyOpInPlace(
        rop, [&] { rop->setAttr(kProcessedMarker, rewriter.getUnitAttr()); });
    return mlir::success();
  }
};

// Single-point anchor at the last iteration of each enclosing loop.
pasm::AnchorRangeAttr build_last_anchor(mlir::MLIRContext *ctx,
                                        mlir::Operation *op,
                                        mlir::FlatSymbolRefAttr id) {
  llvm::SmallVector<uint32_t> idx;
  for (mlir::affine::AffineForOp loop : get_enclosing_loops(op)) {
    int64_t ub =
        loop.hasConstantUpperBound() ? loop.getConstantUpperBound() : 1;
    idx.push_back(static_cast<uint32_t>(ub - 1));
  }
  return make_ir_anchor(ctx, id, idx, idx);
}

// Evt rops feeding `op`, through rops without an evt and through the values
// loops yield.
void evt_producers(mlir::Operation *op,
                   llvm::SmallVectorImpl<mlir::Operation *> &out,
                   std::set<mlir::Operation *> &seen) {
  for (mlir::Value v : op->getOperands()) {
    if (auto loop = v.getDefiningOp<mlir::affine::AffineForOp>()) {
      if (!seen.insert(loop.getOperation()).second) {
        continue;
      }
      mlir::Value yielded = loop.getBody()->getTerminator()->getOperand(
          llvm::cast<mlir::OpResult>(v).getResultNumber());
      auto p = yielded.getDefiningOp<vesyla::drra::RopOp>();
      if (p && p.getEvtAttr()) {
        out.push_back(p);
      } else if (p && seen.insert(p).second) {
        evt_producers(p, out, seen);
      }
      continue;
    }
    auto p = v.getDefiningOp<vesyla::drra::RopOp>();
    if (!p || !seen.insert(p).second) {
      continue;
    }
    if (p.getEvtAttr()) {
      out.push_back(p);
    } else {
      evt_producers(p, out, seen);
    }
  }
}

// DPU confs of one epoch on one DPU, grouped into runs (program order) that
// use the same configuration.
struct DpuSegment {
  int64_t option;
  llvm::SmallVector<vesyla::drra::RopOp> confs;
};


// Puts each distinct DPU configuration of an epoch in its own config register
// (`option`). One configuration: the register the DPU has selected from the
// previous epoch (`selected`), no event. Several: options 0.. in order of first
// use, and an evt on the dpu port with `init_addr` = option before each run,
// after the previous run's last event. Runs alternating in a loop get their
// evts inside it, repeated every pass.
mlir::LogicalResult select_dpu_options(
    pasm::EpochOp epoch,
    std::map<std::tuple<int32_t, int32_t, int32_t>, int64_t> &selected) {
  mlir::MLIRContext *ctx = epoch.getContext();
  std::map<std::tuple<int32_t, int32_t, int32_t>,
           llvm::SmallVector<vesyla::drra::RopOp>>
      by_dpu;
  epoch.walk([&](vesyla::drra::RopOp rop) {
    auto kind = rop->getAttrOfType<mlir::StringAttr>("kind");
    if (!kind || kind.getValue() != "dpu" || !rop.getConfAttr()) {
      return;
    }
    pasm::ResourceAttr res;
    mlir::Attribute attr = rop->getAttr("resource");
    if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
      res = arr.empty() ? nullptr : mlir::dyn_cast<pasm::ResourceAttr>(arr[0]);
    } else {
      res = mlir::dyn_cast_or_null<pasm::ResourceAttr>(attr);
    }
    if (res) {
      by_dpu[{res.getRow(), res.getCol(), res.getSlot()}].push_back(rop);
    }
  });

  mlir::Block &top = epoch.getBody().front();
  mlir::OpBuilder builder(ctx);
  auto set_option = [&](vesyla::drra::RopOp rop, int64_t option) {
    mlir::NamedAttrList conf(rop.getConfAttr());
    conf.set("option", builder.getI32IntegerAttr(option));
    rop.setConfAttr(conf.getDictionary(ctx));
  };
  auto config_key = [&](vesyla::drra::RopOp rop) {
    mlir::NamedAttrList conf(rop.getConfAttr());
    conf.erase("option");
    return conf.getDictionary(ctx);
  };

  for (auto &[key, confs] : by_dpu) {
    llvm::SmallVector<mlir::DictionaryAttr> configs;
    llvm::SmallVector<DpuSegment> segments;
    for (vesyla::drra::RopOp rop : confs) {
      mlir::DictionaryAttr cfg = config_key(rop);
      auto it = llvm::find(configs, cfg);
      int64_t option = it - configs.begin();
      if (it == configs.end()) {
        configs.push_back(cfg);
      }
      if (segments.empty() || segments.back().option != option) {
        segments.push_back({option, {}});
      }
      segments.back().confs.push_back(rop);
    }
    auto [row, col, slot] = key;
    int64_t options = pasm::option_limit_at(row, col, slot);
    if (configs.size() > static_cast<size_t>(options)) {
      return confs.front().emitError()
             << "the DPU at slot " << slot << " needs " << configs.size()
             << " configurations in one epoch, more than its " << options
             << " config registers (NUM_OPTIONS)";
    }
    if (configs.size() == 1) {
      int64_t option = selected.count(key) ? selected[key] : 0;
      for (vesyla::drra::RopOp rop : confs) {
        set_option(rop, option);
      }
      continue;
    }

    // Runs in one top-level nest alternate in every pass of the innermost loop
    // holding them all: their selects go into that loop's body.
    llvm::DenseMap<mlir::Operation *, mlir::Block *> hosts;
    {
      llvm::DenseMap<mlir::Operation *,
                     llvm::SmallVector<mlir::affine::AffineForOp>>
          common;
      llvm::DenseMap<mlir::Operation *, llvm::SmallSetVector<int64_t, 4>> runs;
      for (auto [n, seg] : llvm::enumerate(segments)) {
        for (vesyla::drra::RopOp rop : seg.confs) {
          mlir::Operation *nest = top.findAncestorOpInBlock(*rop);
          runs[nest].insert(n);
          auto loops = get_enclosing_loops(rop);
          auto it = common.find(nest);
          if (it == common.end()) {
            common[nest] = loops;
            continue;
          }
          auto &shared = it->second;
          size_t k = 0;
          while (k < shared.size() && k < loops.size() && shared[k] == loops[k])
            ++k;
          shared.resize(k);
        }
      }
      for (auto &[nest, loops] : common) {
        if (runs[nest].size() > 1 && !loops.empty()) {
          hosts[nest] = loops.back().getBody();
        }
      }
    }

    mlir::OpBuilder cb(ctx);
    cb.setInsertionPoint(top.getTerminator());
    auto cstr = [&](mlir::Location loc, pasm::AnchorRangeAttr src,
                    pasm::AnchorRangeAttr dst, llvm::StringRef rule,
                    int64_t delay) {
      pasm::CstrOp::create(cb, loc, src, dst,
                           pasm::DelayAttr::get(ctx, delay, std::nullopt),
                           cb.getBoolAttr(false))
          ->setAttr("debug", debug_attr(cb, rule));
    };
    // From the events of a nest's last run to its first select one pass
    // later, for every loop around the select.
    auto wrap = [&](vesyla::drra::RopOp first_sel,
                    llvm::ArrayRef<mlir::Operation *> events) {
      auto sel_id = first_sel->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
      for (mlir::Operation *e : events) {
        auto id = e->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
        if (!id) {
          continue;
        }
        auto shared = shared_loops(e, first_sel);
        llvm::SmallVector<bool> reused(shared.size(), true);
        for (size_t d = 0; d < shared.size(); ++d) {
          if (trip_count(shared[d]) < 2) {
            continue;
          }
          cstr(first_sel.getLoc(),
               build_carry(ctx, e, shared, reused, id, Pin::last, d, 0),
               build_carry(ctx, first_sel, shared, reused, sel_id, Pin::first,
                           d, 1),
               "dpu-select-carry", 1);
        }
      }
    };

    llvm::DenseMap<mlir::Attribute, llvm::SmallVector<mlir::Operation *>>
        by_state;
    epoch.walk([&](vesyla::drra::RopOp rop) {
      if (auto state = rop->getAttr("state")) {
        by_state[state].push_back(rop);
      }
    });

    mlir::Operation *prev_at = nullptr;
    mlir::Operation *prev_nest = nullptr;
    vesyla::drra::RopOp first_sel;
    llvm::SmallVector<mlir::Operation *> prev_events;
    for (auto [n, seg] : llvm::enumerate(segments)) {
      llvm::SmallVector<mlir::Operation *> producers;
      std::set<mlir::Operation *> seen;
      for (vesyla::drra::RopOp rop : seg.confs) {
        set_option(rop, seg.option);
        evt_producers(rop, producers, seen);
      }
      mlir::Operation *nest = top.findAncestorOpInBlock(*seg.confs.front());
      mlir::Block *host = hosts.lookup(nest);
      if (!host) {
        host = &top;
      }
      // Before the earliest op in the host block holding the run's first conf,
      // its operands or the clears of its state.
      mlir::Operation *at = host->findAncestorOpInBlock(*seg.confs.front());
      llvm::SmallVector<mlir::Operation *> before(producers);
      for (vesyla::drra::RopOp rop : seg.confs) {
        if (auto state = rop->getAttr("state")) {
          llvm::append_range(before, by_state.lookup(state));
        }
      }
      for (mlir::Operation *p : before) {
        mlir::Operation *a = host->findAncestorOpInBlock(*p);
        if (a && a->isBeforeInBlock(at)) {
          at = a;
        }
      }
      if (prev_at &&
          (prev_at->getBlock() == host ? !prev_at->isBeforeInBlock(at)
                                       : !prev_nest->isBeforeInBlock(nest))) {
        return seg.confs.front().emitError()
               << "DPU slot " << slot
               << ": configurations alternate inside one loop";
      }
      if (first_sel && nest != prev_nest) {
        wrap(first_sel, prev_events);
        first_sel = vesyla::drra::RopOp();
      }
      builder.setInsertionPoint(at);
      std::string name = "dpu_select_" + std::to_string(slot) + "_" +
                         std::to_string(n) + "_" +
                         epoch.getId().str();
      auto sel = vesyla::drra::RopOp::create(
          builder, at->getLoc(), mlir::TypeRange{}, mlir::ValueRange{},
          mlir::DictionaryAttr(),
          builder.getDictionaryAttr(
              {builder.getNamedAttr("port", builder.getI32IntegerAttr(0)),
               builder.getNamedAttr("init_addr",
                                    builder.getI32IntegerAttr(seg.option))}));
      auto sel_id = mlir::FlatSymbolRefAttr::get(ctx, name);
      sel->setAttr("id", sel_id);
      sel->setAttr("kind", builder.getStringAttr("dpu"));
      sel->setAttr("resource", pasm::ResourceAttr::get(ctx, row, col, slot, 0));
      sel->setAttr("uses", builder.getStrArrayAttr({"agu:0"}));
      if (host != &top) {
        // The same register on every pass.
        unsigned dims = get_enclosing_loops(sel).size();
        sel->setAttr("map", mlir::AffineMapAttr::get(mlir::AffineMap::get(
                                dims, 0, mlir::getAffineConstantExpr(0, ctx))));
        if (!first_sel) {
          first_sel = sel;
        }
      }

      // After every event of the previous run: its operands and its results'
      // consumers.
      for (mlir::Operation *e : prev_events) {
        auto id = e->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
        if (id) {
          auto shared = shared_loops(e, sel);
          cstr(sel.getLoc(), build_within(ctx, e, shared, id, Pin::last),
               build_within(ctx, sel, shared, sel_id, Pin::first),
               "dpu-select-after", 1);
        }
      }
      // Before the run's first operand, one cycle more than the switch needs:
      // a mac runs stage 1 every cycle it is selected, and the cycle after the
      // switch belongs to its stage 2.
      for (mlir::Operation *p : producers) {
        auto id = p->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
        if (id) {
          auto shared = shared_loops(sel, p);
          cstr(sel.getLoc(), build_within(ctx, sel, shared, sel_id, Pin::first),
               build_within(ctx, p, shared, id, Pin::first),
               "dpu-select-before", 2);
        }
      }

      prev_at = at;
      prev_nest = nest;
      prev_events.clear();
      std::set<mlir::Operation *> seen_events;
      for (mlir::Operation *p : producers) {
        if (seen_events.insert(p).second) {
          prev_events.push_back(p);
        }
        for (const ReachedEvent &r : reached_events(p)) {
          if (seen_events.insert(r.event).second) {
            prev_events.push_back(r.event);
          }
        }
      }
    }
    if (first_sel) {
      wrap(first_sel, prev_events);
    }
    selected[key] = segments.back().option;
  }
  return mlir::success();
}

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
    // second statement's first loop iteration, within each pass through the
    // loops the two share.
    mlir::MLIRContext *ctx = &getContext();
    std::map<std::tuple<int32_t, int32_t, int32_t>, int64_t> dpu_selected;
    bool failed = false;
    getOperation().walk([&](pasm::EpochOp epoch) {
      if (failed || mlir::failed(select_dpu_options(epoch, dpu_selected))) {
        failed = true;
        return;
      }
      mlir::OpBuilder builder(ctx);
      builder.setInsertionPoint(epoch.getBody().front().getTerminator());
      constrain_stream_steps(epoch, builder);

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
          // Two accesses to the same storage are a memory dependence, not
          // two strangers contending for a slot, and the contention shape is
          // wrong for them rather than merely conservative. Last-to-first says
          // every access of the first happens before every access of the
          // second; for a program that refills the storage on each pass
          // through a loop the two share, that hoists every write ahead of
          // every read, and the storage holds only the last pass's data by the
          // time anything reads it. Anchoring inside the shared loops instead
          // states it per pass, which is what the program means.
          //
          // A pair that shares a slot but no storage is anchored the same way:
          // inside a loop the two share they take turns once per pass, and
          // all-before-all would forbid the program its own interleaving. With
          // no shared loop the anchors collapse to last-to-first.
          mlir::FlatSymbolRefAttr a_storage = storage_of(a);
          mlir::FlatSymbolRefAttr b_storage = storage_of(b);
          bool same_storage = a_storage && a_storage == b_storage;
          llvm::SmallVector<mlir::affine::AffineForOp> shared =
              shared_loops(a, b);
          auto delay_of = [&](mlir::Operation *from, mlir::Operation *to) {
            return pasm::DelayAttr::get(
                ctx, same_storage ? storage_delay(from, to) : 1, std::nullopt);
          };

          pasm::AnchorRangeAttr src =
              build_within(ctx, a, shared, a_id, Pin::last);
          pasm::AnchorRangeAttr dst =
              build_within(ctx, b, shared, b_id, Pin::first);
          pasm::CstrOp::create(builder, a->getLoc(), src, dst, delay_of(a, b),
                         builder.getBoolAttr(false))
              ->setAttr("debug",
                        debug_attr(builder, same_storage
                                                ? "storage-within-pass"
                                                : "contention-within-pass"));

          // The pass-to-pass half of the same dependence. Saying only that a
          // pass's accesses of `a` come before that pass's accesses of `b`
          // leaves the next pass's `a` free to run first and overwrite what
          // this pass's `b` has not read yet -- the schedule satisfies every
          // per-pass constraint and still loses the data. That only arises
          // where the pair meets again, so it is asked of every shared loop
          // that is reused and of no other: a program that addresses somewhere
          // new each pass may overlap them freely.
          llvm::SmallVector<bool> reused =
              reused_loops(a, b, shared, same_storage);
          // A read of `b` followed by the next pass's write of `a`: each row's
          // write waits only for the last read of that row.
          auto row_points =
              same_storage && storage_delay(b, a) == 0
                  ? row_war_points(b, a, shared.size())
                  : decltype(row_war_points(b, a, 0)){};
          // Two io reads in a double-buffered loop (innermost shared loop of
          // two passes, DoubleBufferPass) that share only the external read
          // port: the next pass's `a` may come before this pass's `b` (a row
          // fetch ahead of the previous pass's word read). Ordered two passes
          // apart instead of one, and kept off the cycles they could now
          // share: a(p+1) != b(p), for every point of either.
          auto a_points = inner_points(a, shared.size(), 4);
          auto b_points = inner_points(b, shared.size(), 4);
          if (!same_storage && !shared.empty() &&
              trip_count(shared.back()) == 2 &&
              share_only_external_read(a, b) && a_points && b_points) {
            llvm::SmallVector<bool> two_apart(reused);
            two_apart.back() = false;
            for (size_t carry = 0; carry + 1 < shared.size(); ++carry) {
              if (!reused[carry] || trip_count(shared[carry]) < 2) {
                continue;
              }
              pasm::CstrOp::create(
                  builder, b->getLoc(),
                  build_carry(ctx, b, shared, two_apart, b_id, Pin::last,
                              carry, 0),
                  build_carry(ctx, a, shared, two_apart, a_id, Pin::first,
                              carry, 1),
                  delay_of(b, a), builder.getBoolAttr(false))
                  ->setAttr("debug", debug_attr(builder, "contention-carry-2"));
            }
            auto zero = pasm::DelayAttr::get(ctx, 0, std::nullopt);
            for (size_t carry = 0; carry < shared.size(); ++carry) {
              if (!reused[carry] || trip_count(shared[carry]) < 2) {
                continue;
              }
              for (auto &bp : *b_points) {
                for (auto &ap : *a_points) {
                  pasm::CstrOp::create(
                      builder, b->getLoc(),
                      build_carry(ctx, b, shared, reused, b_id, Pin::last,
                                  carry, 0, bp),
                      build_carry(ctx, a, shared, reused, a_id, Pin::first,
                                  carry, 1, ap),
                      zero, builder.getBoolAttr(true))
                      ->setAttr("debug",
                                debug_attr(builder, "contention-not-same-cycle"));
                }
              }
            }
            continue;
          }
          for (size_t carry = 0; carry < shared.size(); ++carry) {
            if (!reused[carry] || trip_count(shared[carry]) < 2) {
              continue;
            }
            for (auto &[read_at, write_at] : row_points) {
              pasm::CstrOp::create(
                  builder, b->getLoc(),
                  build_carry(ctx, b, shared, reused, b_id, Pin::last, carry,
                              0, read_at),
                  build_carry(ctx, a, shared, reused, a_id, Pin::first, carry,
                              1, write_at),
                  delay_of(b, a), builder.getBoolAttr(false))
                  ->setAttr("debug", debug_attr(builder, "storage-carry-row"));
            }
            if (!row_points.empty()) {
              continue;
            }
            pasm::AnchorRangeAttr carry_src = build_carry(
                ctx, b, shared, reused, b_id, Pin::last, carry, 0);
            pasm::AnchorRangeAttr carry_dst = build_carry(
                ctx, a, shared, reused, a_id, Pin::first, carry, 1);
            pasm::CstrOp::create(builder, b->getLoc(), carry_src, carry_dst,
                           delay_of(b, a), builder.getBoolAttr(false))
                ->setAttr("debug",
                          debug_attr(builder, same_storage
                                                  ? "storage-carry"
                                                  : "contention-carry"));
          }
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
        // Each is kept with the rule that made it a use. A config in a loop nest
        // prefers a use in that nest, else the first anywhere.
        llvm::SmallVector<std::pair<mlir::Operation *, llvm::StringRef>>
            first_uses;
        std::set<mlir::Operation *> seen;
        mlir::Block &top = epoch.getBody().front();
        mlir::Operation *cfg_nest = top.findAncestorOpInBlock(*cfg);
        for (const auto &key : cfg_keys) {
          mlir::Operation *first = nullptr;
          for (mlir::Operation *use : statements) {
            std::set<std::tuple<int32_t, int32_t, int32_t>> use_keys;
            collect_resource_keys(use, use_keys);
            if (!use_keys.count(key)) {
              continue;
            }
            if (!first) {
              first = use;
            }
            if (cfg_nest != cfg && top.findAncestorOpInBlock(*use) == cfg_nest) {
              first = use;
              break;
            }
          }
          if (first && seen.insert(first).second) {
            first_uses.push_back({first, "config-first-use"});
          }
        }

        // A config with no event of its own is used by whatever it is fed. It
        // sits on the data path with nothing on its own resource to say when
        // that starts, so the search above can come back empty, and the config
        // is then free to land after the data has gone by. The events that
        // produce its operands are its first uses.
        // Producers without an evt (another DPU mode, a loop's result) are
        // followed back to the events that feed them.
        auto cfg_rop = mlir::dyn_cast<vesyla::drra::RopOp>(cfg);
        if (cfg_rop && !cfg_rop.getEvtAttr()) {
          llvm::SmallVector<mlir::Operation *> producers;
          std::set<mlir::Operation *> walked;
          evt_producers(cfg, producers, walked);
          for (mlir::Operation *producer : producers) {
            if (seen.insert(producer).second) {
              first_uses.push_back({producer, "config-before-operand"});
            }
          }
        }

        for (auto [use, rule] : first_uses) {
          auto use_id =
              mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(use->getAttr("id"));
          if (!use_id) {
            continue;
          }
          auto src = make_ir_anchor(ctx, conf_symbol(cfg, cfg_id), {}, {});
          pasm::AnchorRangeAttr dst = build_first_anchor(ctx, use, use_id);
          auto delay_attr = pasm::DelayAttr::get(ctx, 1, std::nullopt);
          pasm::CstrOp::create(builder, cfg->getLoc(), src, dst, delay_attr,
                         builder.getBoolAttr(false))
              ->setAttr("debug", debug_attr(builder, rule));
        }
      }
    });
    if (failed) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::conversion::drra_to_pasm
