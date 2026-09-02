#ifndef __VESYLA_TRANSFORMATION_DSE_ARCHITECTURE_HPP__
#define __VESYLA_TRANSFORMATION_DSE_ARCHITECTURE_HPP__

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "json/json.hpp"

#include <string>

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
  // Elaborated parameters of this instance (RF_DEPTH, WORD_BITWIDTH, ...).
  // Binding will need them to know a register file's capacity; nothing reads
  // them yet.
  nlohmann::json parameters;
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
