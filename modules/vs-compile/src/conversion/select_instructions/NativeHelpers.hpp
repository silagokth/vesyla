#ifndef __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_NATIVE_HELPERS_HPP__
#define __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_NATIVE_HELPERS_HPP__

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

// Predicates and IR utilities the matcher leans on. Kept out of the matcher so
// the walk itself stays about structure, and ordinary enough to test directly.
namespace vesyla {
namespace sel {

// True when `acc` is the loop-carried accumulator of an enclosing affine.for --
// a region iter_args block argument. This is the program-side half of the
// accumulator rule: the resource writes its running total to a register, the
// program carries it as an SSA value, and the two are the same thing.
bool is_iter_args_carry(mlir::Value acc);

// True when `op` produces a value that seeds an affine.for iter_args. This is
// what stops a reset pattern -- a bare constant, once its store is accounted
// for -- from claiming every constant in the program.
bool feeds_iter_args_init(mlir::Operation *op);

// Number of affine.for ops enclosing `op`.
unsigned enclosing_loop_depth(mlir::Operation *op);

// Best-effort lift of the affine access map, describing the address the AGU
// sweeps:
//   - outside any affine.for: no map (returns null; the access is not iterated).
//   - inside affine.for(s): reuse the map of the affine.apply that feeds an
//     index operand, if any; otherwise fall back to identity onto the innermost
//     loop dim.
// A heuristic that matches the shapes in the reference IR; the exact
// multi-index / symbol cases need revisiting once run end to end. Unused while
// the pass only annotates, but it is what the drra emission will need.
mlir::AffineMapAttr lift_affine_map(mlir::Operation *access);

} // namespace sel
} // namespace vesyla

#endif // __VESYLA_CONVERSION_SELECT_INSTRUCTIONS_NATIVE_HELPERS_HPP__
