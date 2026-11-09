#ifndef __VESYLA_SUPPORT_LOOP_SHAPE_HPP__
#define __VESYLA_SUPPORT_LOOP_SHAPE_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace vesyla::util {

// The trip count of an affine.for as the AGU runs it (AGU Slope Levels): a
// constant, or `iter + slope * c` with `c` the index of the enclosing loop
// `follows` loops up (1: the parent), for a loop whose upper bound is
// `slope * c + iter` after NormalizeLoops. `iter` is the count in the first
// pass of the followed loop.
struct LoopShape {
  int64_t iter = 0;
  int64_t slope = 0;
  unsigned follows = 0; // 0: a constant trip count
  bool constant() const { return follows == 0; }
};

// The shape of `loop`, or nothing for bounds this cannot read: a lower bound
// other than 0, a step other than 1, an upper bound over a symbol or over more
// than one induction variable.
std::optional<LoopShape> loop_shape(mlir::affine::AffineForOp loop);

// The enclosing affine.for loops of `op`, outermost first.
llvm::SmallVector<mlir::affine::AffineForOp>
enclosing_loops(mlir::Operation *op);

// The loop whose index the trip count of `loop` follows, or null.
mlir::affine::AffineForOp followed_loop(mlir::affine::AffineForOp loop);

// Trips of `loop` when its enclosing loops (outermost first) stand at `idx`.
// A loop this cannot read counts as one trip.
int64_t trips_at(mlir::affine::AffineForOp loop, llvm::ArrayRef<int64_t> idx);

// The largest trip count of `loop` over every pass of its enclosing loops.
int64_t max_trips(mlir::affine::AffineForOp loop);

// The trip count of `loop` in the last pass of every enclosing loop.
int64_t trips_at_last(mlir::affine::AffineForOp loop);

// The last index of every loop of a nest (outermost first) when all of them
// are in their last pass.
llvm::SmallVector<int64_t>
last_indices(llvm::ArrayRef<mlir::affine::AffineForOp> loops);

// Calls `visit` at every point of the iteration domain of a nest (outermost
// first), in program order. False when the domain has more than `limit`
// points or a loop cannot be read.
bool for_each_point(llvm::ArrayRef<mlir::affine::AffineForOp> loops,
                    int64_t limit,
                    llvm::function_ref<void(llvm::ArrayRef<int64_t>)> visit);

} // namespace vesyla::util

#endif // __VESYLA_SUPPORT_LOOP_SHAPE_HPP__
