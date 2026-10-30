#ifndef __VESYLA_TRANSFORMATION_DSE_ARCHITECTURE_HPP__
#define __VESYLA_TRANSFORMATION_DSE_ARCHITECTURE_HPP__

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "json/json.hpp"

#include <string>
#include <utility>

namespace vesyla {
namespace transformation {
namespace dse {

// One physical resource instance in the fabric: what exists at (row, col,
// slot), as the architecture file says.
struct ResourceInstance {
  // Resource kind, spelled the way the component library spells it ("rf",
  // "dpu", "io", ...). A drra.rop's `kind` attribute is matched against this.
  std::string kind;
  int row = 0;
  int col = 0;
  // First slot the instance occupies; it covers `slot` .. `slot + size - 1`.
  int slot = 0;
  int size = 1;
  bool io_input = false;
  bool io_output = false;
  // The component's category ("memory", ...), "uncategorized" when it has none.
  std::string category;
  // Elaborated parameters of this instance (RF_DEPTH, WORD_BITWIDTH, ...).
  nlohmann::json parameters;
  // The instructions it takes, as (name, port): one entry per port key of an
  // instruction with a `port` segment, port -1 for one without.
  llvm::SmallVector<std::pair<std::string, int>> instructions;

  bool is_memory() const { return category == "memory"; }
  bool has_instruction(llvm::StringRef name, int port) const;
  // A memory's capacity in words (RF_DEPTH), and the words in one of its rows
  // (a bulk transfer: BULK_BITWIDTH / WORD_BITWIDTH). 0 when not given.
  int depth() const;
  int row_words() const;
  int depth_rows() const;
};

// The set of resources the design may use -- the allocation.
//
// Allocation is an input for now rather than a decision: the architecture file
// passed to the executable fixes which resources exist and where they sit, and
// this class only reads it. When allocation becomes something to search over,
// it is this table that the search produces and everything downstream of it
// stays as it is.
class Architecture {
public:
  // Build from the arch json already loaded into the Config singleton.
  //
  // This is the *elaborated* architecture -- what vs-component writes to
  // work/system/arch/arch.json -- not the source form found under test/. The
  // elaborated form lists cell instances with their coordinates and gives every
  // resource a concrete slot; the source form lists cell *definitions* and has
  // neither. Traversal is guarded throughout, so a source-form file (or any
  // other unexpected shape) is reported against `diag_op` rather than throwing
  // out of nlohmann::json.
  static mlir::FailureOr<Architecture> from_config(mlir::Operation *diag_op);

  llvm::ArrayRef<ResourceInstance> instances() const { return instances_; }
  const ResourceInstance &at(unsigned index) const { return instances_[index]; }
  unsigned size() const { return instances_.size(); }

  // Indices of every instance of `kind`, in the order the architecture file
  // lists them (row-major over cells, then slot order within a cell).
  llvm::SmallVector<unsigned> instances_of_kind(llvm::StringRef kind) const;

  // Indices of every memory instance, smallest depth first, then in file
  // order.
  llvm::SmallVector<unsigned> memory_instances() const;

  int rows() const { return rows_; }
  int cols() const { return cols_; }

private:
  llvm::SmallVector<ResourceInstance> instances_;
  int rows_ = 0;
  int cols_ = 0;
};

} // namespace dse
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_DSE_ARCHITECTURE_HPP__
