#ifndef __VESYLA_TRANSFORMATION_DSE_MEMORY_MAP_HPP__
#define __VESYLA_TRANSFORMATION_DSE_MEMORY_MAP_HPP__

#include "Architecture.hpp"
#include "ConflictGraph.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vesyla {
namespace transformation {
namespace bind_resources {

// Where each storage lives in the memories: an instance and a range of rows,
// held from the storage's first write to its last read.
//
// TIME. Binding runs before scheduling, so there are no cycles to go by. Time
// is program order instead, the way liveness reads it for every other
// resource: a point is one top-level operation of an epoch, numbered across
// the epochs in order, and an operation inside an affine.for takes the point
// of its outermost loop -- the loop runs as a whole, so a storage touched
// anywhere in it is held for all of it. Reusing rows a storage has freed is
// then safe without anything new downstream: across epochs the epochs run in
// order, and within one CreateConstraintsPass already orders every two
// operations on one instance, the last of the earlier before the first of the
// later.
//
// LIFETIME. One interval per storage, [first, last]: from its first write
// until its last read. A storage whose first read can see a word nothing wrote
// before it relies on the memory's reset value 0 (dot drains a row of which it
// wrote only word 0), so it is held from the start of the program, on rows
// nothing used before. The words written before the first read are worked out
// from each write's map and constant loop bounds; a write that cannot be read
// that way counts as covering nothing. A write at the same point as the first
// read, earlier in program order, counts for the first pass of the loops the
// two share (a buffer filled and read inside one loop). A storage that is
// written again after it was last read still holds its rows in between.
//
// ROWS. A storage takes whole rows (a bulk transfer each), so its base is a
// row and every bulk access to it stays aligned.

// A stretch of program points, both ends included.
struct Interval {
  unsigned first = 0;
  unsigned last = 0;
  bool overlaps(const Interval &other) const {
    return first <= other.last && other.first <= last;
  }
};

// The program points of a module: every top-level operation of every epoch,
// in program order.
class ProgramPoints {
public:
  // Fails, with a diagnostic, on an epoch inside a loop: its operations would
  // run once per iteration, which no interval over the epochs describes.
  static mlir::FailureOr<ProgramPoints> build(mlir::ModuleOp module);

  // The point `op` runs at, or nothing when it is in no epoch.
  std::optional<unsigned> point(mlir::Operation *op) const;
  // The last point of the program.
  unsigned last() const { return names_.empty() ? 0 : names_.size() - 1; }
  // "<epoch>.<n>": the n-th top-level operation of the epoch.
  std::string name(unsigned point) const;

private:
  llvm::DenseMap<mlir::Operation *, unsigned> points_;
  llvm::SmallVector<std::string> names_;
};

// The lifetime of the storage node `node`, `words` long, from its accesses: an
// access with a result reads the storage, one without writes it.
Interval storage_lifetime(const ConflictGraph &graph, unsigned node,
                          int64_t words, const ProgramPoints &points);

// Which rows of which memory hold which storage, and when.
class MemoryMap {
public:
  struct Entry {
    mlir::FlatSymbolRefAttr storage;
    Interval live;
  };

  explicit MemoryMap(const Architecture &arch);

  // Whether anything is held on `instance` at some point of `live`.
  bool busy(unsigned instance, const Interval &live) const;
  // The lowest base row from which `rows` rows of `instance` are free over
  // `live`, or nothing.
  std::optional<int> fit(unsigned instance, int rows,
                         const Interval &live) const;
  void place(unsigned instance, int base, int rows, const Entry &entry);
  // What holds `instance` at some point of `live`, for a diagnostic:
  // "rows 0-1 hold @x [rb0.0, rb1.2]; ...".
  std::string holders(unsigned instance, const Interval &live,
                      const ProgramPoints &points) const;

  // One line per row of every memory, with what it holds and when.
  void write(llvm::raw_ostream &os, const ProgramPoints &points) const;

private:
  const Architecture &arch_;
  // instance -> row -> entries
  std::map<unsigned, std::vector<std::vector<Entry>>> rows_;
};

} // namespace bind_resources
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_DSE_MEMORY_MAP_HPP__
