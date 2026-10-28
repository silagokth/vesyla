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
#include "conversion/select_instructions/NativeHelpers.hpp"

#include <map>
#include <tuple>

namespace vesyla::transformation::stage_io {
#define GEN_PASS_DEF_STAGEIOPASS
#include "transformation/stage_io/Passes.hpp.inc"

namespace {

using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::affine::AffineForOp;

// A word-by-word access to an io buffer: the map's first result picks the
// row, the second the word in it.
struct IoAccess {
  mlir::Operation *op;
  bool write;
  mlir::Value buffer;
  AffineMap map;
  llvm::SmallVector<mlir::Value> operands;

  AffineExpr row() const { return map.getResult(0); }
  AffineExpr word() const { return map.getResult(1); }
};

std::optional<IoAccess> as_io_access(mlir::Operation *op) {
  IoAccess access;
  if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(op)) {
    access = {op, false, load.getMemRef(), load.getAffineMap(),
              llvm::to_vector(load.getMapOperands())};
  } else if (auto store = llvm::dyn_cast<mlir::affine::AffineStoreOp>(op)) {
    access = {op, true, store.getMemRef(), store.getAffineMap(),
              llvm::to_vector(store.getMapOperands())};
  } else {
    return std::nullopt;
  }
  if (!vesyla::sel::is_io_buffer(access.buffer) ||
      access.map.getNumResults() != 2) {
    return std::nullopt;
  }
  return access;
}

llvm::SmallVector<IoAccess> collect_io_accesses(mlir::ModuleOp module) {
  llvm::SmallVector<IoAccess> accesses;
  module.walk([&](mlir::Operation *op) {
    if (auto access = as_io_access(op)) {
      accesses.push_back(*access);
    }
  });
  return accesses;
}

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

// Splits `loop` (0 to n, n a multiple of w) into a loop over rows and the
// original loop, now 0 to w, over the words of a row: x = row * w + word.
// iter_args move to the outer loop and are threaded through the inner one.
void strip_mine(AffineForOp loop, int64_t w) {
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

// The loops whose counters `e` reads.
llvm::SmallVector<AffineForOp> loops_of(const IoAccess &access, AffineExpr e) {
  llvm::SmallVector<AffineForOp> loops;
  for (auto [pos, operand] : llvm::enumerate(access.operands)) {
    if (pos < access.map.getNumDims() && e.isFunctionOfDim(pos)) {
      loops.push_back(mlir::affine::getForInductionVarOwner(operand));
    }
  }
  return loops;
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

// Why `access` cannot be staged, or empty when it can.
std::string unstageable(const IoAccess &access, mlir::Operation *point,
                        int64_t w) {
  if (auto loop = loop_to_split(access, w);
      loop && splits_a_reduction(*loop, w)) {
    return "its row changes inside a reduction loop, and splitting that loop "
           "per row is not supported yet (C frontend G18)";
  }
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
  for (AffineForOp row : loops_of(access, access.row())) {
    if (llvm::is_contained(loops_of(access, access.word()), row)) {
      return access.write
                 ? "its row changes with the word index"
                 : "its row changes with the word index (a sliding window, "
                   "C frontend G17)";
    }
  }
  for (AffineForOp loop : loops_of(access, access.word())) {
    if (loop.getOperation() != point && loop->isProperAncestor(point)) {
      return "its row changes inside the loop over its words";
    }
  }
  if (!access.write) {
    return "";
  }
  // A write has to cover the whole row: its word is the counter of a loop
  // over exactly the row's words.
  auto word = llvm::dyn_cast<mlir::AffineDimExpr>(access.word());
  int64_t upper;
  if (!word ||
      !is_constant_loop(mlir::affine::getForInductionVarOwner(
                            access.operands[word.getPosition()]),
                        upper) ||
      upper != w) {
    return "it writes only part of the row; partial row writes are not "
           "supported yet (C frontend G16)";
  }
  return "";
}

// Accesses to the same row of the same buffer at the same place share one
// staging buffer.
struct Group {
  bool write;
  mlir::Value buffer;
  mlir::Operation *point;
  AffineMap row;
  llvm::SmallVector<mlir::Value> rowOperands;
  llvm::SmallVector<IoAccess> accesses;
};

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
      }
    }

    // Check everything before changing anything, so one run reports every
    // access that cannot be staged.
    llvm::SmallVector<Group> groups;
    bool failed = false;
    for (const IoAccess &access : collect_io_accesses(module)) {
      int64_t w = row_words(access.buffer);
      mlir::Operation *point = staging_point(access);
      std::string why = unstageable(access, point, w);
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
               g.point == point && g.row == row && g.rowOperands == rowOperands;
      });
      if (same == groups.end()) {
        groups.push_back(
            {access.write, access.buffer, point, row, rowOperands, {}});
        same = std::prev(groups.end());
      }
      same->accesses.push_back(access);
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
