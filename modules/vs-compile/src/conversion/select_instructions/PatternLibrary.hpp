#ifndef __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PATTERN_LIBRARY_HPP__
#define __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PATTERN_LIBRARY_HPP__

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace vesyla {
namespace conversion {
namespace select_instructions {

// One selectable pattern: a single func.func from a resource's <kind>.mlir in
// the component library.
//
// The function's block arguments are the wildcards, its body is the shape to
// match, and `benefit` ranks it against the others. The `conf` / `evt`
// dictionaries say which instruction a match becomes; they are validated
// against the resource's own isa.json at load time.
struct Pattern {
  std::string kind; // enclosing module symbol: "dpu", "rf", "io", ...
  std::string name; // function symbol: "mac", "word_read", ...
  int64_t benefit = 0;

  // The segments the replacement's drra.rop carries, kept here for validation
  // and reporting. Null when the rop declares no such field. At least one of
  // the two is present.
  mlir::DictionaryAttr conf;
  mlir::DictionaryAttr evt;

  // The shape to look for, from the resource's @match module, and what it turns
  // into, from @replace. They share a signature, so replacement argument i
  // stands for whatever match argument i bound to -- MLIR drops SSA names at
  // parse time, so position is what links the two.
  mlir::func::FuncOp match;
  mlir::func::FuncOp replace;

  // The op the match is rooted at: the defining op of the returned value, or
  // for a void function the last body op that is not the terminator and not
  // the accumulator write-back. The replacement's results stand in for this
  // op's results.
  mlir::Operation *root = nullptr;

  // Set when the function has an accumulator argument -- a rank-0 memref
  // argument, which is how a resource's own register is written.
  // `accumulatorStore` is the write-back, which the accumulator rule consumes
  // instead of matching.
  mlir::BlockArgument accumulator;
  mlir::Operation *accumulatorStore = nullptr;
  bool readsAccumulator = false;

  // True for a pattern that writes the accumulator without reading it: @rst,
  // not @mac. Its root is whatever produces the initial value, so a match has
  // to be an affine.for iter_args initialiser -- otherwise every zero constant
  // in the program would look like a reset.
  bool initialisesAccumulator() const {
    return accumulator && accumulatorStore && !readsAccumulator;
  }

  // True for the io and iosram_* resources, whose patterns only apply to a
  // memref allocated as an IO buffer.
  bool guardedByBufferRole() const;
};

// Loads every resource's functional description out of the installed component
// library and turns it into a ranked list of patterns.
class PatternLibrary {
public:
  // Reads <componentPath>/resources/*/<kind>.mlir plus each sibling isa.json.
  // Fails on a file that does not parse, a function with no emission attribute,
  // or an emission value that does not validate against isa.json.
  mlir::LogicalResult load(llvm::StringRef componentPath, mlir::MLIRContext *ctx,
                           llvm::raw_ostream &diag);

  // Matchable patterns grouped by benefit, highest tier first. Patterns within
  // a tier are tried together and are unordered relative to each other.
  const std::vector<std::vector<const Pattern *>> &tiers() const {
    return tiers_;
  }

  // Loaded but deliberately never matched, with the reason. Reported once so a
  // resource author can see that a function was read and skipped rather than
  // silently ignored.
  const std::vector<std::pair<std::string, std::string>> &skipped() const {
    return skipped_;
  }

private:
  mlir::LogicalResult loadResource(llvm::StringRef mlirPath,
                                   llvm::StringRef isaPath,
                                   mlir::MLIRContext *ctx,
                                   llvm::raw_ostream &diag);

  // Owns the parsed pattern modules; Pattern refers into them.
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules_;
  std::vector<std::unique_ptr<Pattern>> patterns_;
  std::vector<std::vector<const Pattern *>> tiers_;
  std::vector<std::pair<std::string, std::string>> skipped_;
};

} // namespace select_instructions
} // namespace conversion
} // namespace vesyla

#endif // __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_PATTERN_LIBRARY_HPP__
