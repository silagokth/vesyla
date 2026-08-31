#ifndef __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_MATCHER_HPP__
#define __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_MATCHER_HPP__

#include "PatternLibrary.hpp"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace vesyla {
namespace conversion {
namespace select_instructions {

struct MatchResult {
  // Pattern value -> the program value it bound to. A wildcard that appears
  // twice in the pattern has to bind to the same program value both times.
  llvm::DenseMap<mlir::Value, mlir::Value> bindings;
  // The program ops this match covers, root last.
  llvm::SmallVector<mlir::Operation *, 8> cone;
};

// Walks a pattern function's body and a fragment of the program at the same
// time, one operand pair at a time, starting from the pattern's root.
//
// The two sides are not identical IR and are not meant to be: the pattern
// describes a resource written for the reference configuration, the program
// describes a computation. Five relaxations bridge that, and they are the whole
// substance of the matcher:
//
//   1. Memory ops form two classes -- every read matches every read, every
//      write every write -- and what discriminates is the type of the value
//      moved, not the op name.
//   2. The memref operand of a memory op is a wildcard: its rank, shape,
//      element type and address space are not matched.
//   3. Index operands are wildcards and the pattern's own address arithmetic
//      is never walked into. The AGU computes the address.
//   4. A read of the pattern's rank-0 accumulator argument matches an
//      affine.for iter_args carry -- the register in the pattern and the SSA
//      loop carry in the program are the same thing.
//   5. An io / iosram pattern only matches where the program says it is
//      touching an IO buffer.
class Matcher {
public:
  explicit Matcher(const Pattern &pattern) : pattern_(pattern) {}

  // True when the pattern matches the program fragment rooted at `candidate`.
  // On success `result` holds the bindings and the covered ops.
  bool match(mlir::Operation *candidate, MatchResult &result) const;

private:
  bool matchOp(mlir::Operation *patternOp, mlir::Operation *inputOp,
               MatchResult &result) const;
  bool matchValue(mlir::Value patternValue, mlir::Value inputValue,
                  MatchResult &result) const;
  bool matchMemref(mlir::Value patternMemref, mlir::Value inputMemref,
                   MatchResult &result) const;

  // True when `patternValue` is a load of this pattern's accumulator register.
  bool isAccumulatorRead(mlir::Value patternValue) const;

  const Pattern &pattern_;
};

} // namespace select_instructions
} // namespace conversion
} // namespace vesyla

#endif // __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_MATCHER_HPP__
