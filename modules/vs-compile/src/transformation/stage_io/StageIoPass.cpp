#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "StageIoPass.hpp"
#include "IoAccess.hpp"
#include "conversion/select_instructions/NativeHelpers.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <tuple>

namespace vesyla::transformation::stage_io {
#define GEN_PASS_DEF_STAGEIOPASS
#include "transformation/stage_io/Passes.hpp.inc"

namespace {

using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::affine::AffineForOp;
using namespace vesyla::transformation::stage_io;

// The words in an io row: its memref is <rows x words x elem>.
int64_t row_words(mlir::Value buffer) {
  return llvm::cast<mlir::MemRefType>(buffer.getType()).getShape()[1];
}

// `e` with `d<pos> floordiv w` replaced by `floorDiv`, `d<pos> mod w` by `mod`
// and any other `d<pos>` by `dim`.
AffineExpr substitute(AffineExpr e, unsigned pos, int64_t w,
                      AffineExpr floorDiv, AffineExpr mod, AffineExpr dim) {
  if (auto bin = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(e)) {
    auto lhs = llvm::dyn_cast<mlir::AffineDimExpr>(bin.getLHS());
    auto rhs = llvm::dyn_cast<mlir::AffineConstantExpr>(bin.getRHS());
    if (lhs && rhs && lhs.getPosition() == pos && rhs.getValue() == w) {
      if (e.getKind() == mlir::AffineExprKind::FloorDiv) {
        return floorDiv;
      }
      if (e.getKind() == mlir::AffineExprKind::Mod) {
        return mod;
      }
    }
    return mlir::getAffineBinaryOpExpr(
        e.getKind(), substitute(bin.getLHS(), pos, w, floorDiv, mod, dim),
        substitute(bin.getRHS(), pos, w, floorDiv, mod, dim));
  }
  if (auto d = llvm::dyn_cast<mlir::AffineDimExpr>(e)) {
    if (d.getPosition() == pos) {
      return dim;
    }
  }
  return e;
}

bool is_constant_loop(AffineForOp loop, int64_t &upper) {
  if (!loop.hasConstantLowerBound() || loop.getConstantLowerBound() != 0 ||
      !loop.hasConstantUpperBound() || loop.getStepAsInt() != 1) {
    return false;
  }
  upper = loop.getConstantUpperBound();
  return true;
}

// Rebuilds an affine access with a new map and operands, keeping its
// attributes (the id).
void rebuild_access(mlir::Operation *op, AffineMap map,
                    llvm::ArrayRef<mlir::Value> operands) {
  mlir::OpBuilder builder(op);
  mlir::Operation *rebuilt = nullptr;
  if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
    rebuilt = mlir::affine::AffineLoadOp::create(
        builder, op->getLoc(), load.getMemRef(), map, operands);
  } else if (auto store = llvm::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
    rebuilt = mlir::affine::AffineStoreOp::create(
        builder, op->getLoc(), store.getValueToStore(), store.getMemRef(), map,
        operands);
  } else if (auto load = llvm::dyn_cast<mlir::affine::AffineVectorLoadOp>(op)) {
    rebuilt = mlir::affine::AffineVectorLoadOp::create(
        builder, op->getLoc(), load.getVectorType(), load.getMemRef(), map,
        operands);
  } else if (auto store =
                 llvm::dyn_cast<mlir::affine::AffineVectorStoreOp>(op)) {
    rebuilt = mlir::affine::AffineVectorStoreOp::create(
        builder, op->getLoc(), store.getValueToStore(), store.getMemRef(), map,
        operands);
  }
  rebuilt->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  op->replaceAllUsesWith(rebuilt);
  op->erase();
}

// Every affine access inside `loop` that is indexed by its counter.
llvm::SmallVector<mlir::Operation *> accesses_using(AffineForOp loop) {
  mlir::Value iv = loop.getInductionVar();
  llvm::SmallVector<mlir::Operation *> found;
  loop.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp,
                  mlir::affine::AffineVectorLoadOp,
                  mlir::affine::AffineVectorStoreOp>(op) &&
        llvm::is_contained(op->getOperands(), iv)) {
      found.push_back(op);
    }
  });
  return found;
}

AffineMap access_map(mlir::Operation *op) {
  if (auto read = llvm::dyn_cast<mlir::affine::AffineReadOpInterface>(op)) {
    return read.getAffineMap();
  }
  return llvm::cast<mlir::affine::AffineWriteOpInterface>(op).getAffineMap();
}

llvm::SmallVector<mlir::Value> access_operands(mlir::Operation *op) {
  if (auto read = llvm::dyn_cast<mlir::affine::AffineReadOpInterface>(op)) {
    return llvm::to_vector(read.getMapOperands());
  }
  return llvm::to_vector(
      llvm::cast<mlir::affine::AffineWriteOpInterface>(op).getMapOperands());
}

// A loop of at most w iterations: `x floordiv w` is 0 and `x mod w` is x.
void fold_short_loop(AffineForOp loop, int64_t w) {
  mlir::MLIRContext *ctx = loop.getContext();
  for (mlir::Operation *op : accesses_using(loop)) {
    AffineMap map = access_map(op);
    llvm::SmallVector<mlir::Value> operands = access_operands(op);
    unsigned pos =
        llvm::find(operands, loop.getInductionVar()) - operands.begin();
    llvm::SmallVector<AffineExpr> results;
    for (AffineExpr e : map.getResults()) {
      results.push_back(mlir::simplifyAffineExpr(
          substitute(e, pos, w, mlir::getAffineConstantExpr(0, ctx),
                     mlir::getAffineDimExpr(pos, ctx),
                     mlir::getAffineDimExpr(pos, ctx)),
          map.getNumDims(), map.getNumSymbols()));
    }
    rebuild_access(
        op, AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, ctx),
        operands);
  }
}

// A copy of `loop` for its last `rem` iterations, right after it, counting
// from 0: x = base + counter. `loop` keeps the rest.
void split_remainder(AffineForOp loop, int64_t rem) {
  mlir::MLIRContext *ctx = loop.getContext();
  int64_t base = loop.getConstantUpperBound() - rem;
  mlir::OpBuilder builder(loop);
  builder.setInsertionPointAfter(loop);
  auto tail = llvm::cast<AffineForOp>(builder.clone(*loop));
  tail.setConstantUpperBound(rem);
  loop.setConstantUpperBound(base);

  mlir::Value counter = tail.getInductionVar();
  llvm::SmallVector<mlir::Operation *> accesses = accesses_using(tail);
  builder.setInsertionPointToStart(tail.getBody());
  mlir::Value whole;
  for (mlir::OpOperand &use : llvm::make_early_inc_range(counter.getUses())) {
    if (llvm::is_contained(accesses, use.getOwner()) ||
        (whole && use.getOwner() == whole.getDefiningOp())) {
      continue;
    }
    if (!whole) {
      whole = mlir::affine::AffineApplyOp::create(
          builder, loop.getLoc(),
          AffineMap::get(1, 0, mlir::getAffineDimExpr(0, ctx) + base), counter);
    }
    use.set(whole);
  }
  for (mlir::Operation *op : accesses) {
    AffineMap map = access_map(op);
    llvm::SmallVector<mlir::Value> operands = access_operands(op);
    unsigned pos = llvm::find(operands, counter) - operands.begin();
    AffineExpr dim = mlir::getAffineDimExpr(pos, ctx);
    llvm::SmallVector<AffineExpr> results;
    for (AffineExpr e : map.getResults()) {
      results.push_back(mlir::simplifyAffineExpr(e.replace(dim, dim + base),
                                                 map.getNumDims(),
                                                 map.getNumSymbols()));
    }
    rebuild_access(
        op, AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, ctx),
        operands);
  }
}

// Splits `loop` (0 to n) into a loop over rows and the original loop, now 0 to
// w, over the words of a row: x = row * w + word. What is left of n after whole
// rows runs as a loop of its own. iter_args move to the outer loop and are
// threaded through the inner one.
void strip_mine(AffineForOp loop, int64_t w) {
  if (int64_t rem = loop.getConstantUpperBound() % w) {
    split_remainder(loop, rem);
  }
  mlir::MLIRContext *ctx = loop.getContext();
  mlir::OpBuilder builder(loop);
  int64_t upper = loop.getConstantUpperBound();
  auto outer = AffineForOp::create(builder, loop.getLoc(), 0, upper / w, 1,
                                   loop.getInits());
  mlir::Value row = outer.getInductionVar();
  mlir::Value word = loop.getInductionVar();

  loop.getResults().replaceAllUsesWith(outer.getResults());
  mlir::Block *body = outer.getBody();
  if (loop.getInits().empty()) {
    loop->moveBefore(body->getTerminator());
  } else {
    loop->moveBefore(body, body->end());
    builder.setInsertionPointToEnd(body);
    mlir::affine::AffineYieldOp::create(builder, loop.getLoc(),
                                        loop.getResults());
    loop.getInitsMutable().assign(outer.getRegionIterArgs());
  }
  loop.setConstantUpperBound(w);

  // Uses of the counter outside an access -- an affine.apply, a bound --
  // still mean x, so they get x back.
  llvm::SmallVector<mlir::Operation *> accesses = accesses_using(loop);
  builder.setInsertionPointToStart(loop.getBody());
  mlir::Value whole;
  for (mlir::OpOperand &use : llvm::make_early_inc_range(word.getUses())) {
    if (llvm::is_contained(accesses, use.getOwner())) {
      continue;
    }
    if (!whole) {
      AffineExpr x =
          mlir::getAffineDimExpr(0, ctx) * w + mlir::getAffineDimExpr(1, ctx);
      whole = mlir::affine::AffineApplyOp::create(builder, loop.getLoc(),
                                                  AffineMap::get(2, 0, x),
                                                  mlir::ValueRange{row, word});
    }
    use.set(whole);
  }

  for (mlir::Operation *op : accesses) {
    AffineMap map = access_map(op);
    llvm::SmallVector<mlir::Value> operands = access_operands(op);
    unsigned pos = llvm::find(operands, word) - operands.begin();
    unsigned dims = map.getNumDims();
    // The row counter becomes a new last dim; symbols keep their positions.
    AffineExpr rowDim = mlir::getAffineDimExpr(dims, ctx);
    AffineExpr wordDim = mlir::getAffineDimExpr(pos, ctx);
    llvm::SmallVector<AffineExpr> results;
    for (AffineExpr e : map.getResults()) {
      results.push_back(mlir::simplifyAffineExpr(
          substitute(e, pos, w, rowDim, wordDim, rowDim * w + wordDim),
          dims + 1, map.getNumSymbols()));
    }
    operands.insert(operands.begin() + dims, row);
    rebuild_access(op,
                   AffineMap::get(dims + 1, map.getNumSymbols(), results, ctx),
                   operands);
  }
}

// The loop to strip-mine (or fold) for `access`, if its row is `x floordiv w`
// and its word `x mod w` for some loop counter x.
std::optional<AffineForOp> loop_to_split(const IoAccess &access, int64_t w) {
  mlir::MLIRContext *ctx = access.op->getContext();
  for (auto [pos, operand] : llvm::enumerate(access.operands)) {
    AffineForOp loop = mlir::affine::getForInductionVarOwner(operand);
    if (!loop || pos >= access.map.getNumDims()) {
      continue;
    }
    AffineExpr x = mlir::getAffineDimExpr(pos, ctx);
    if (access.word() != x % w) {
      continue;
    }
    AffineExpr rest =
        substitute(access.row(), pos, w, mlir::getAffineConstantExpr(0, ctx),
                   mlir::getAffineConstantExpr(0, ctx), x);
    if (rest.isFunctionOfDim(pos)) {
      continue;
    }
    int64_t upper;
    if (is_constant_loop(loop, upper) && (upper <= w || upper % w == 0)) {
      return loop;
    }
  }
  return std::nullopt;
}

// A reduction loop (one carrying iter_args) that would have to be split: the
// accumulation would then run over two loops, which the DPU's rst/mac
// patterns do not reset per result (C frontend G18). Folding a short one is
// fine -- it stays one loop.
bool splits_a_reduction(AffineForOp loop, int64_t w) {
  return !loop.getInits().empty() && loop.getConstantUpperBound() > w;
}

// Calls `visit` with `e` at every point of the loops it reads. False when they
// are not known at compile time.
bool for_each_value(const IoAccess &access, AffineExpr e,
                    llvm::function_ref<void(int64_t)> visit) {
  llvm::SmallVector<AffineExpr> dims;
  for (unsigned pos = 0; pos < access.map.getNumDims(); ++pos) {
    dims.push_back(mlir::getAffineConstantExpr(0, access.op->getContext()));
  }
  std::function<bool(unsigned)> walk = [&](unsigned pos) {
    if (pos == dims.size()) {
      auto value = llvm::dyn_cast<mlir::AffineConstantExpr>(
          mlir::simplifyAffineExpr(e.replaceDims(dims), 0, 0));
      if (!value) {
        return false;
      }
      visit(value.getValue());
      return true;
    }
    if (!e.isFunctionOfDim(pos)) {
      return walk(pos + 1);
    }
    AffineForOp loop = mlir::affine::getForInductionVarOwner(access.operands[pos]);
    if (!loop.hasConstantBounds()) {
      return false;
    }
    for (int64_t i = loop.getConstantLowerBound();
         i < loop.getConstantUpperBound(); i += loop.getStepAsInt()) {
      dims[pos] = mlir::getAffineConstantExpr(i, access.op->getContext());
      if (!walk(pos + 1)) {
        return false;
      }
    }
    return true;
  };
  return walk(0);
}

// The words of its row a write stores, with bit n for word n. Nothing when
// they are not known at compile time or leave the row.
std::optional<int64_t> written_words(const IoAccess &access, int64_t w) {
  int64_t mask = 0;
  bool inside = true;
  if (!for_each_value(access, access.word(), [&](int64_t word) {
        inside &= word >= 0 && word < w;
        if (inside) {
          mask |= int64_t{1} << word;
        }
      }) ||
      !inside) {
    return std::nullopt;
  }
  return mask;
}

// Row and word read the same counter: a walk over the words that crosses rows.
bool crosses_rows(const IoAccess &access) {
  auto rows = loops_of(access, access.row());
  return llvm::any_of(loops_of(access, access.word()), [&](AffineForOp loop) {
    return llvm::is_contained(rows, loop);
  });
}

// The most io rows a flat buffer holds: the largest RF, rf_4row.
constexpr int64_t MAX_FLAT_ROWS = 4;

// Where an access that crosses rows reads: `rows` io rows from row `lo` on,
// copied side by side into a flat buffer in the body of `level` (null: ahead of
// all its loops), which it reads at `index`.
struct FlatPlan {
  AffineForOp level;
  AffineExpr lo;
  int64_t rows;
  AffineExpr index;
};

// The outermost level whose iterations fit a flat buffer. Its counters must
// not appear in the index: they pick the rows, the buffer never moves.
std::optional<FlatPlan> plan_flat(const IoAccess &access, int64_t w) {
  mlir::MLIRContext *ctx = access.op->getContext();
  llvm::SmallVector<AffineForOp> levels{AffineForOp()};
  llvm::SmallVector<AffineForOp> around;
  for (auto loop = access.op->getParentOfType<AffineForOp>(); loop;
       loop = loop->getParentOfType<AffineForOp>()) {
    around.insert(around.begin(), loop);
  }
  levels.append(around);

  unsigned dims = access.map.getNumDims();
  for (AffineForOp level : levels) {
    // The rows start where the loops inside the level start.
    llvm::SmallVector<AffineExpr> starts;
    llvm::SmallVector<unsigned> fixed;
    for (unsigned pos = 0; pos < dims; ++pos) {
      AffineForOp loop =
          mlir::affine::getForInductionVarOwner(access.operands[pos]);
      if (level && (loop == level || loop->isProperAncestor(level))) {
        starts.push_back(mlir::getAffineDimExpr(pos, ctx));
        fixed.push_back(pos);
      } else if (loop.hasConstantLowerBound()) {
        starts.push_back(
            mlir::getAffineConstantExpr(loop.getConstantLowerBound(), ctx));
      } else {
        return std::nullopt;
      }
    }
    AffineExpr lo = mlir::simplifyAffineExpr(access.row().replaceDims(starts),
                                             dims, 0);
    AffineExpr index = mlir::simplifyAffineExpr(
        (access.row() - lo) * w + access.word(), dims, 0);
    if (!index.isPureAffine() ||
        llvm::any_of(fixed, [&](unsigned pos) {
          return index.isFunctionOfDim(pos);
        })) {
      continue;
    }
    int64_t low = 0, high = 0;
    if (!for_each_value(access, index, [&](int64_t i) {
          low = std::min(low, i);
          high = std::max(high, i);
        })) {
      return std::nullopt;
    }
    if (low >= 0 && high / w < MAX_FLAT_ROWS) {
      return FlatPlan{level, lo, high / w + 1, index};
    }
  }
  return std::nullopt;
}

// The loop to strip-mine so an access that crosses rows fits a flat buffer.
std::optional<AffineForOp> window_loop(const IoAccess &access, int64_t w) {
  auto rows = loops_of(access, access.row());
  std::optional<AffineForOp> outermost;
  for (AffineForOp loop : loops_of(access, access.word())) {
    int64_t upper;
    if (llvm::is_contained(rows, loop) && loop.getInits().empty() &&
        is_constant_loop(loop, upper) && upper > w &&
        (!outermost || loop->isProperAncestor(*outermost))) {
      outermost = loop;
    }
  }
  return outermost;
}

// Where an access's row is staged: the op, directly in the body of the
// innermost loop the row depends on, that holds the access. Without such a
// loop, the outermost loop around the access (or the access itself).
mlir::Operation *staging_point(const IoAccess &access) {
  AffineForOp innermost;
  for (AffineForOp loop : loops_of(access, access.row())) {
    if (!innermost || innermost->isProperAncestor(loop)) {
      innermost = loop;
    }
  }
  if (innermost) {
    return innermost.getBody()->findAncestorOpInBlock(*access.op);
  }
  mlir::Operation *point = access.op;
  for (mlir::Operation *parent = access.op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (llvm::isa<AffineForOp>(parent)) {
      point = parent;
    }
  }
  return point;
}

// Why `access` cannot go through a buffer at all, or empty when it can.
std::string unbufferable(const IoAccess &access) {
  for (auto [pos, operand] : llvm::enumerate(access.operands)) {
    if (pos >= access.map.getNumDims() ||
        !mlir::affine::getForInductionVarOwner(operand)) {
      return "its index uses a value that is not a loop counter";
    }
  }
  auto id =
      access.buffer.getDefiningOp()->getAttrOfType<mlir::FlatSymbolRefAttr>(
          "id");
  if (!access.write && id.getValue() == "output_buffer") {
    return "it reads the output buffer";
  }
  if (access.write && id.getValue() == "input_buffer") {
    return "it writes the input buffer";
  }
  return "";
}

// Why `access` cannot be staged a row at a time, or empty when it can.
std::string unstageable(const IoAccess &access, mlir::Operation *point,
                        int64_t w) {
  // A read fetches its row again on every pass; a write would overwrite the
  // row's other words.
  for (AffineForOp loop : loops_of(access, access.word())) {
    if (access.write && loop.getOperation() != point &&
        loop->isProperAncestor(point)) {
      return "its row changes inside the loop over its words";
    }
  }
  if (access.write && !written_words(access, w)) {
    return "the words it writes are not known at compile time";
  }
  return "";
}

// Accesses to the same row of the same buffer in one block share one staging
// buffer, read ahead of the first and flushed after the last.
struct Group {
  bool write;
  mlir::Value buffer;
  mlir::Operation *point;
  AffineMap row;
  llvm::SmallVector<mlir::Value> rowOperands;
  llvm::SmallVector<IoAccess> accesses;
  int64_t words = 0; // a write's words, bit n = word n
};

// Accesses that cross rows from the same first row in one block share one flat
// buffer.
struct FlatGroup {
  bool write;
  mlir::Value buffer;
  mlir::Operation *point;
  AffineMap lo;
  llvm::SmallVector<mlir::Value> loOperands;
  llvm::SmallVector<std::pair<IoAccess, AffineExpr>, 1> accesses; // and index
  int64_t rows = 0;
  llvm::SmallVector<int64_t> words; // per row, a write's words, bit n = word n
};

// The op directly in the body of `level` (null: of the epoch) that holds `op`.
mlir::Operation *op_in_level(AffineForOp level, mlir::Operation *op) {
  if (level) {
    return level.getBody()->findAncestorOpInBlock(*op);
  }
  mlir::Operation *top = op;
  for (auto loop = op->getParentOfType<AffineForOp>(); loop;
       loop = loop->getParentOfType<AffineForOp>()) {
    top = loop;
  }
  return top;
}

class StageIoPass : public impl::StageIoPassBase<StageIoPass> {
public:
  using impl::StageIoPassBase<StageIoPass>::StageIoPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    // Strip-mine (or fold) until no access picks its row and word out of one
    // counter. Each change rebuilds accesses, so collect afresh after it.
    for (bool changed = true; changed;) {
      changed = false;
      for (const IoAccess &access : collect_io_accesses(module)) {
        int64_t w = row_words(access.buffer);
        if (auto loop = loop_to_split(access, w);
            loop && !splits_a_reduction(*loop, w)) {
          if (loop->getConstantUpperBound() <= w) {
            fold_short_loop(*loop, w);
          } else {
            strip_mine(*loop, w);
          }
          changed = true;
          break;
        }
        if (crosses_rows(access) && unbufferable(access).empty() &&
            !plan_flat(access, w)) {
          if (auto loop = window_loop(access, w)) {
            strip_mine(*loop, w);
            changed = true;
            break;
          }
        }
      }
    }

    // Check everything before changing anything, so one run reports every
    // access that cannot be staged.
    llvm::SmallVector<Group> groups;
    llvm::SmallVector<FlatGroup, 0> flats;
    bool failed = false;
    for (const IoAccess &access : collect_io_accesses(module)) {
      int64_t w = row_words(access.buffer);
      if (crosses_rows(access) && unbufferable(access).empty()) {
        std::optional<FlatPlan> plan = plan_flat(access, w);
        if (!plan) {
          access.op->emitError("stage-io: this word-by-word access to an io "
                               "buffer needs more rows than a flat buffer "
                               "holds, or an index that is not linear");
          failed = true;
          continue;
        }
        AffineMap lo = AffineMap::get(access.map.getNumDims(),
                                      access.map.getNumSymbols(), plan->lo, ctx);
        llvm::SmallVector<mlir::Value> loOperands = access.operands;
        mlir::affine::canonicalizeMapAndOperands(&lo, &loOperands);
        mlir::Operation *point = op_in_level(plan->level, access.op);
        auto same = llvm::find_if(flats, [&](const FlatGroup &g) {
          return g.write == access.write && g.buffer == access.buffer &&
                 g.point->getBlock() == point->getBlock() && g.lo == lo &&
                 g.loOperands == loOperands;
        });
        if (same == flats.end()) {
          flats.push_back({access.write, access.buffer, point, lo, loOperands});
          same = std::prev(flats.end());
        }
        if (access.write) {
          same->point = point;
        }
        same->accesses.push_back({access, plan->index});
        same->rows = std::max(same->rows, plan->rows);
        same->words.resize(same->rows);
        if (access.write) {
          for_each_value(access, plan->index, [&](int64_t i) {
            same->words[i / w] |= int64_t{1} << (i % w);
          });
        }
        continue;
      }
      mlir::Operation *point = staging_point(access);
      std::string why = unbufferable(access);
      if (why.empty()) {
        why = unstageable(access, point, w);
      }
      if (!why.empty()) {
        access.op->emitError("stage-io: this word-by-word access to an io "
                             "buffer cannot be staged: ")
            << why;
        failed = true;
        continue;
      }
      AffineMap row =
          AffineMap::get(access.map.getNumDims(), access.map.getNumSymbols(),
                         access.row(), ctx);
      llvm::SmallVector<mlir::Value> rowOperands = access.operands;
      mlir::affine::canonicalizeMapAndOperands(&row, &rowOperands);
      auto same = llvm::find_if(groups, [&](const Group &g) {
        return g.write == access.write && g.buffer == access.buffer &&
               g.point->getBlock() == point->getBlock() && g.row == row &&
               g.rowOperands == rowOperands;
      });
      if (same == groups.end()) {
        groups.push_back(
            {access.write, access.buffer, point, row, rowOperands, {}});
        same = std::prev(groups.end());
      }
      same->accesses.push_back(access);
      if (access.write) {
        same->point = point;
        same->words |= *written_words(access, w);
      }
    }
    if (failed) {
      return signalPassFailure();
    }

    // A row written word for word with the words of a row read at the same
    // place (a copy: `out[r][e] = in[r][e]`) needs no second buffer and no
    // scalar copy: the read's staging buffer is written out as it is.
    llvm::DenseMap<size_t, size_t> copiedFrom; // write group -> read group
    for (auto [w, written] : llvm::enumerate(groups)) {
      if (!written.write) {
        continue;
      }
      std::optional<size_t> source;
      bool copy = true;
      for (const IoAccess &store : written.accesses) {
        mlir::Value value =
            llvm::cast<mlir::affine::AffineStoreOp>(store.op).getValueToStore();
        mlir::Operation *load = value.getDefiningOp();
        auto read = llvm::find_if(groups, [&](const Group &g) {
          return !g.write && g.point == written.point &&
                 llvm::any_of(g.accesses, [&](const IoAccess &a) {
                   return a.op == load && a.word() == store.word() &&
                          a.operands == store.operands;
                 });
        });
        size_t r = read - groups.begin();
        if (read == groups.end() || !value.hasOneUse() ||
            (source && *source != r)) {
          copy = false;
          break;
        }
        source = r;
      }
      // Every word the read stages has to go to this write and nowhere else.
      if (copy && source &&
          groups[*source].accesses.size() == written.accesses.size()) {
        copiedFrom[w] = *source;
      }
    }
    llvm::DenseSet<size_t> copySources;
    for (auto [w, r] : copiedFrom) {
      copySources.insert(r);
    }

    // Staging buffers live with the module's other buffers, after the last.
    mlir::Operation *lastAlloc = nullptr;
    for (auto alloc : module.getOps<mlir::memref::AllocOp>()) {
      lastAlloc = alloc;
    }

    // Reads first, so a copy's write finds its read's buffer.
    llvm::SmallVector<mlir::Value> stages(groups.size());
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> lastFlush;
    llvm::SmallVector<mlir::Operation *> copyOps;
    for (bool writes : {false, true}) {
      for (auto [g, group] : llvm::enumerate(groups)) {
        if (group.write != writes) {
          continue;
        }
        auto ioType = llvm::cast<mlir::MemRefType>(group.buffer.getType());
        int64_t w = ioType.getShape()[1];
        mlir::Type elem = ioType.getElementType();
        auto vectorType = mlir::VectorType::get({w}, elem);
        mlir::OpBuilder builder(ctx);

        auto copied = copiedFrom.find(g);
        if (copied != copiedFrom.end()) {
          stages[g] = stages[copied->second];
        } else {
          builder.setInsertionPointAfter(lastAlloc);
          auto stage = mlir::memref::AllocOp::create(
              builder, group.point->getLoc(), mlir::MemRefType::get({w}, elem));
          lastAlloc = stage;
          stages[g] = stage;
        }
        mlir::Value stage = stages[g];

        AffineMap ioMap = AffineMap::get(
            group.row.getNumDims(), group.row.getNumSymbols(),
            {group.row.getResult(0), mlir::getAffineConstantExpr(0, ctx)}, ctx);
        AffineMap stageMap =
            AffineMap::get(0, 0, mlir::getAffineConstantExpr(0, ctx));
        mlir::Location loc = group.accesses.front().op->getLoc();

        if (!group.write) {
          builder.setInsertionPoint(group.point);
          auto fetched = mlir::affine::AffineVectorLoadOp::create(
              builder, loc, vectorType, group.buffer, ioMap, group.rowOperands);
          mlir::affine::AffineVectorStoreOp::create(builder, loc, fetched,
                                                    stage, stageMap, {});
        } else {
          mlir::Operation *after = lastFlush.lookup(group.point);
          builder.setInsertionPointAfter(after ? after : group.point);
          auto staged = mlir::affine::AffineVectorLoadOp::create(
              builder, loc, vectorType, stage, stageMap, {});
          auto flushed = mlir::affine::AffineVectorStoreOp::create(
              builder, loc, staged, group.buffer, ioMap, group.rowOperands);
          if (group.words != (int64_t{1} << w) - 1) {
            flushed->setAttr("wr_mask", builder.getI32IntegerAttr(group.words));
          }
          lastFlush[group.point] = flushed;
        }

        // A copy's scalar loads and stores go away with it; erased below,
        // stores before the loads they use.
        if (copied != copiedFrom.end() || copySources.contains(g)) {
          for (IoAccess &access : group.accesses) {
            copyOps.push_back(access.op);
          }
          continue;
        }

        // The scalar access itself now goes to the staging buffer, at the
        // same word.
        for (IoAccess &access : group.accesses) {
          AffineMap word =
              AffineMap::get(access.map.getNumDims(),
                             access.map.getNumSymbols(), access.word(), ctx);
          mlir::OpBuilder at(access.op);
          mlir::Operation *rebuilt;
          if (access.write) {
            rebuilt = mlir::affine::AffineStoreOp::create(
                at, access.op->getLoc(),
                llvm::cast<mlir::affine::AffineStoreOp>(access.op)
                    .getValueToStore(),
                stage, word, access.operands);
          } else {
            rebuilt = mlir::affine::AffineLoadOp::create(
                at, access.op->getLoc(), stage, word, access.operands);
          }
          rebuilt->setDiscardableAttrs(
              access.op->getDiscardableAttrDictionary());
          access.op->replaceAllUsesWith(rebuilt);
          access.op->erase();
        }
      }
    }
    // Flat buffers: the rows side by side, each access at its index. A vector
    // access to a buffer addresses it in rows.
    for (FlatGroup &group : flats) {
      auto ioType = llvm::cast<mlir::MemRefType>(group.buffer.getType());
      int64_t w = ioType.getShape()[1];
      mlir::Type elem = ioType.getElementType();
      auto vectorType = mlir::VectorType::get({w}, elem);
      mlir::Location loc = group.accesses.front().first.op->getLoc();
      unsigned dims = group.lo.getNumDims();
      unsigned syms = group.lo.getNumSymbols();
      AffineExpr zero = mlir::getAffineConstantExpr(0, ctx);

      mlir::OpBuilder builder(ctx);
      builder.setInsertionPointAfter(lastAlloc);
      auto stage = mlir::memref::AllocOp::create(
          builder, group.point->getLoc(),
          mlir::MemRefType::get({group.rows * w}, elem));
      lastAlloc = stage;

      if (!group.write) {
        builder.setInsertionPoint(group.point);
        llvm::SmallVector<mlir::Value> operands = group.loOperands;
        AffineExpr row = group.lo.getResult(0);
        AffineMap stageMap = AffineMap::get(0, 0, zero);
        llvm::SmallVector<mlir::Value> stageOperands;
        if (group.rows > 1) {
          auto rows = AffineForOp::create(builder, loc, 0, group.rows);
          builder.setInsertionPointToStart(rows.getBody());
          operands.insert(operands.begin() + dims, rows.getInductionVar());
          row = row + mlir::getAffineDimExpr(dims, ctx);
          dims += 1;
          stageMap = AffineMap::getMultiDimIdentityMap(1, ctx);
          stageOperands.push_back(rows.getInductionVar());
        }
        auto fetched = mlir::affine::AffineVectorLoadOp::create(
            builder, loc, vectorType, group.buffer,
            AffineMap::get(dims, syms, {row, zero}, ctx), operands);
        mlir::affine::AffineVectorStoreOp::create(builder, loc, fetched, stage,
                                                  stageMap, stageOperands);
      } else {
        mlir::Operation *after = lastFlush.lookup(group.point);
        builder.setInsertionPointAfter(after ? after : group.point);
        for (int64_t r = 0; r < group.rows; ++r) {
          auto staged = mlir::affine::AffineVectorLoadOp::create(
              builder, loc, vectorType, stage,
              AffineMap::get(0, 0, mlir::getAffineConstantExpr(r, ctx)), {});
          auto flushed = mlir::affine::AffineVectorStoreOp::create(
              builder, loc, staged, group.buffer,
              AffineMap::get(dims, syms, {group.lo.getResult(0) + r, zero},
                             ctx),
              group.loOperands);
          if (group.words[r] != (int64_t{1} << w) - 1) {
            flushed->setAttr("wr_mask",
                             builder.getI32IntegerAttr(group.words[r]));
          }
          lastFlush[group.point] = flushed;
        }
      }

      for (auto &[access, index] : group.accesses) {
        AffineMap map = AffineMap::get(access.map.getNumDims(),
                                       access.map.getNumSymbols(), index, ctx);
        mlir::OpBuilder at(access.op);
        mlir::Operation *rebuilt;
        if (access.write) {
          rebuilt = mlir::affine::AffineStoreOp::create(
              at, access.op->getLoc(),
              llvm::cast<mlir::affine::AffineStoreOp>(access.op)
                  .getValueToStore(),
              stage, map, access.operands);
        } else {
          rebuilt = mlir::affine::AffineLoadOp::create(
              at, access.op->getLoc(), stage, map, access.operands);
        }
        rebuilt->setDiscardableAttrs(access.op->getDiscardableAttrDictionary());
        access.op->replaceAllUsesWith(rebuilt);
        access.op->erase();
      }
    }

    // Stores were collected after the loads they use, so backwards each op
    // is unused by the time it is erased: a copy's loads feed only its stores.
    for (mlir::Operation *op : llvm::reverse(copyOps)) {
      op->erase();
    }

    // A copy can leave the loop over its words empty. Several copies can
    // share that loop (swap), so each loop is looked at once.
    llvm::SetVector<mlir::Operation *> copyPoints;
    for (auto [w, r] : copiedFrom) {
      copyPoints.insert(groups[w].point);
    }
    for (mlir::Operation *point : copyPoints) {
      auto loop = llvm::dyn_cast<AffineForOp>(point);
      if (loop && loop.getNumResults() == 0 &&
          llvm::hasSingleElement(loop.getBody()->getOperations())) {
        loop->erase();
      }
    }
  }
};

} // namespace
} // namespace vesyla::transformation::stage_io
