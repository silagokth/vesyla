#include "MemoryMap.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <functional>

#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

namespace vesyla {
namespace transformation {
namespace dse {

mlir::FailureOr<ProgramPoints> ProgramPoints::build(mlir::ModuleOp module) {
  ProgramPoints points;
  bool failed = false;
  module.walk<mlir::WalkOrder::PreOrder>([&](pasm::EpochOp epoch) {
    if (epoch->getParentOfType<mlir::affine::AffineForOp>()) {
      epoch.emitError("design-space-exploration: an epoch inside a loop runs "
                      "once per iteration, which the storage lifetimes cannot "
                      "describe");
      failed = true;
      return;
    }
    unsigned index = 0;
    for (mlir::Operation &op : epoch.getBody().front().without_terminator()) {
      points.points_[&op] = points.names_.size();
      points.names_.push_back(epoch.getId().str() + "." +
                              std::to_string(index++));
    }
  });
  if (failed) {
    return mlir::failure();
  }
  return points;
}

std::optional<unsigned> ProgramPoints::point(mlir::Operation *op) const {
  for (; op; op = op->getParentOp()) {
    auto found = points_.find(op);
    if (found != points_.end()) {
      return found->second;
    }
  }
  return std::nullopt;
}

std::string ProgramPoints::name(unsigned point) const {
  return point < names_.size() ? names_[point] : std::to_string(point);
}

namespace {

// The value of `expr` with dims `dims`, or nothing for what it cannot evaluate.
std::optional<int64_t> evaluate(mlir::AffineExpr expr,
                                llvm::ArrayRef<int64_t> dims) {
  if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    return constant.getValue();
  }
  if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dim.getPosition() >= dims.size()) {
      return std::nullopt;
    }
    return dims[dim.getPosition()];
  }
  auto bin = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!bin) {
    return std::nullopt;
  }
  std::optional<int64_t> lhs = evaluate(bin.getLHS(), dims);
  std::optional<int64_t> rhs = evaluate(bin.getRHS(), dims);
  if (!lhs || !rhs) {
    return std::nullopt;
  }
  switch (bin.getKind()) {
  case mlir::AffineExprKind::Add:
    return *lhs + *rhs;
  case mlir::AffineExprKind::Mul:
    return *lhs * *rhs;
  case mlir::AffineExprKind::FloorDiv:
    return *rhs > 0 ? llvm::divideFloorSigned(*lhs, *rhs)
                    : std::optional<int64_t>();
  case mlir::AffineExprKind::CeilDiv:
    return *rhs > 0 ? llvm::divideCeilSigned(*lhs, *rhs)
                    : std::optional<int64_t>();
  case mlir::AffineExprKind::Mod:
    return *rhs > 0 ? llvm::mod(*lhs, *rhs) : std::optional<int64_t>();
  default:
    return std::nullopt;
  }
}

// Mark the words the write `op` covers over all iterations of its loops.
// False when it cannot tell: loops without constant bounds, a map whose dims
// are not the enclosing loops, or more iterations than are worth walking.
bool mark_written(mlir::Operation *op, llvm::MutableArrayRef<bool> written) {
  llvm::SmallVector<mlir::affine::AffineForOp> loops;
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      if (!loop.hasConstantBounds()) {
        return false;
      }
      loops.insert(loops.begin(), loop);
    }
  }

  int64_t width = 1;
  if (op->getNumOperands() > 0) {
    if (auto vector =
            mlir::dyn_cast<mlir::VectorType>(op->getOperand(0).getType())) {
      width = vector.getNumElements();
    }
  }
  mlir::AffineMap map;
  if (auto attr = op->getAttrOfType<mlir::AffineMapAttr>("map")) {
    map = attr.getValue();
    if (map.getNumResults() != 1 || map.getNumSymbols() != 0 ||
        map.getNumDims() != loops.size()) {
      return false;
    }
  }

  constexpr int64_t kMaxIterations = 1 << 16;
  int64_t iterations = 0;
  llvm::SmallVector<int64_t> dims(loops.size());
  std::function<bool(unsigned)> walk = [&](unsigned depth) -> bool {
    if (depth == loops.size()) {
      if (++iterations > kMaxIterations) {
        return false;
      }
      std::optional<int64_t> address =
          map ? evaluate(map.getResult(0), dims) : int64_t(0);
      if (!address) {
        return false;
      }
      for (int64_t word = *address * width;
           word < (*address + 1) * width && word >= 0 &&
           word < static_cast<int64_t>(written.size());
           ++word) {
        written[word] = true;
      }
      return true;
    }
    mlir::affine::AffineForOp loop = loops[depth];
    for (int64_t i = loop.getConstantLowerBound();
         i < loop.getConstantUpperBound(); i += loop.getStepAsInt()) {
      dims[depth] = i;
      if (!walk(depth + 1)) {
        return false;
      }
    }
    return true;
  };
  return walk(0);
}

} // namespace

Interval storage_lifetime(const ConflictGraph &graph, unsigned node,
                          int64_t words, const ProgramPoints &points) {
  std::optional<unsigned> first_write, first_read, last_write, last_read;
  bool outside = false;
  for (mlir::Operation *op : graph.ops(node)) {
    std::optional<unsigned> at = points.point(op);
    if (!at) {
      outside = true;
      continue;
    }
    bool read = op->getNumResults() > 0;
    std::optional<unsigned> &first = read ? first_read : first_write;
    std::optional<unsigned> &last = read ? last_read : last_write;
    first = first ? std::min(*first, *at) : *at;
    last = last ? std::max(*last, *at) : *at;
  }
  // An access no point covers (a program without epochs) could run at any
  // time, so the storage is held for the whole program.
  if (outside) {
    return {0, points.last()};
  }
  // Held from its first write only if the writes before its first read cover
  // every word; otherwise the read sees reset values.
  bool covered = first_write.has_value();
  if (covered && first_read) {
    llvm::SmallVector<bool> written(std::max<int64_t>(words, 0), false);
    for (mlir::Operation *op : graph.ops(node)) {
      std::optional<unsigned> at = points.point(op);
      if (op->getNumResults() == 0 && at && *at < *first_read &&
          !mark_written(op, written)) {
        covered = false;
      }
    }
    covered = covered && llvm::all_of(written, [](bool word) { return word; });
  }
  Interval live;
  live.first = covered ? *first_write : 0;
  live.last = last_read ? *last_read : last_write.value_or(live.first);
  live.last = std::max(live.last, live.first);
  return live;
}

MemoryMap::MemoryMap(const Architecture &arch) : arch_(arch) {
  for (unsigned instance : arch.memory_instances()) {
    rows_[instance].resize(std::max(arch.at(instance).depth_rows(), 0));
  }
}

bool MemoryMap::busy(unsigned instance, const Interval &live) const {
  auto found = rows_.find(instance);
  if (found == rows_.end()) {
    return false;
  }
  for (const std::vector<Entry> &row : found->second) {
    for (const Entry &entry : row) {
      if (entry.live.overlaps(live)) {
        return true;
      }
    }
  }
  return false;
}

std::optional<int> MemoryMap::fit(unsigned instance, int rows,
                                  const Interval &live) const {
  auto found = rows_.find(instance);
  if (found == rows_.end()) {
    return std::nullopt;
  }
  const std::vector<std::vector<Entry>> &table = found->second;
  auto free = [&](int row) {
    return llvm::none_of(table[row], [&](const Entry &entry) {
      return entry.live.overlaps(live);
    });
  };
  for (int base = 0; base + rows <= static_cast<int>(table.size()); ++base) {
    bool all_free = true;
    for (int row = base; row < base + rows && all_free; ++row) {
      all_free = free(row);
    }
    if (all_free) {
      return base;
    }
  }
  return std::nullopt;
}

void MemoryMap::place(unsigned instance, int base, int rows,
                      const Entry &entry) {
  std::vector<std::vector<Entry>> &table = rows_[instance];
  for (int row = base; row < base + rows; ++row) {
    table[row].push_back(entry);
  }
}

std::string MemoryMap::holders(unsigned instance, const Interval &live,
                               const ProgramPoints &points) const {
  auto found = rows_.find(instance);
  if (found == rows_.end()) {
    return "";
  }
  // One clause per storage, with the rows it holds there.
  std::vector<std::pair<const Entry *, std::pair<int, int>>> seen;
  const std::vector<std::vector<Entry>> &table = found->second;
  for (int row = 0; row < static_cast<int>(table.size()); ++row) {
    for (const Entry &entry : table[row]) {
      if (!entry.live.overlaps(live)) {
        continue;
      }
      auto same = llvm::find_if(seen, [&](const auto &clause) {
        return clause.first->storage == entry.storage;
      });
      if (same == seen.end()) {
        seen.push_back({&entry, {row, row}});
      } else {
        same->second.second = row;
      }
    }
  }
  std::string text;
  for (const auto &[entry, rows] : seen) {
    if (!text.empty()) {
      text += "; ";
    }
    text += rows.first == rows.second
                ? "row " + std::to_string(rows.first)
                : "rows " + std::to_string(rows.first) + "-" +
                      std::to_string(rows.second);
    text += " hold @" + entry->storage.getValue().str() + " [" +
            points.name(entry->live.first) + ", " +
            points.name(entry->live.last) + "]";
  }
  return text;
}

void MemoryMap::write(llvm::raw_ostream &os,
                      const ProgramPoints &points) const {
  for (const auto &[instance, table] : rows_) {
    const ResourceInstance &memory = arch_.at(instance);
    os << memory.kind << " (" << memory.row << "," << memory.col << ","
       << memory.slot << ")\n";
    for (unsigned row = 0; row < table.size(); ++row) {
      os << "  row " << row << ":";
      if (table[row].empty()) {
        os << " free";
      }
      for (const Entry &entry : table[row]) {
        os << " @" << entry.storage.getValue() << " ["
           << points.name(entry.live.first) << ", "
           << points.name(entry.live.last) << "]";
      }
      os << "\n";
    }
  }
}

} // namespace dse
} // namespace transformation
} // namespace vesyla
