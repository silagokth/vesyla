#ifndef __VESYLA_TRANSFORMATION_DSE_STRATEGY_HPP__
#define __VESYLA_TRANSFORMATION_DSE_STRATEGY_HPP__

#include "Architecture.hpp"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace vesyla {
namespace transformation {
namespace dse {

// Binding: which instance of the allocated resources each operation runs on.
//
// `scope` is one pasm.epoch -- the region downstream scheduling treats as a
// unit -- or the module itself when the program has no epochs. A binder writes
// its decision as the `resource` attribute of every drra.rop it claims, either
// a single pasm::ResourceAttr or an array laid out [results..., operands...],
// which is what GenerateIcdepPass and DrraToPasmPass read.
class Binder {
public:
  virtual ~Binder() = default;
  virtual mlir::LogicalResult bind(mlir::Operation *scope,
                                   const Architecture &arch) = 0;
};

// Scheduling: where each operation sits relative to the others.
//
// This is not cycle assignment. ScheduleEpochPass below already solves exact
// timing against each resource's timing model; what is wanted here is only the
// relative order, and only for one reason -- it is what says which operations
// could go on the same instance. Two operations may share a resource exactly
// when the schedule keeps them apart, so the relative schedule is the input
// binding needs, which is why a scheduler runs before a binder rather than
// after one.
//
// The result is annotated onto the operations themselves; there is nothing to
// write out beside the module.
//
// No implementation yet.
class Scheduler {
public:
  virtual ~Scheduler() = default;
  virtual mlir::LogicalResult schedule(mlir::Operation *scope,
                                       const Architecture &arch) = 0;
};

// The one binder that exists so far. It is a placeholder, not an algorithm:
// with no schedule to say which operations are kept apart, it cannot tell a
// shareable instance from a contended one, so every operation goes to the first
// instance of its kind and two accesses to different register files land on the
// same slot. What it buys is a module that carries a resource on every
// operation, which is what the rest of the pipeline -- and the check at the end
// of this pass -- needs to have something to work on meanwhile.
std::unique_ptr<Binder> create_placeholder_binder();

} // namespace dse
} // namespace transformation
} // namespace vesyla

#endif // __VESYLA_TRANSFORMATION_DSE_STRATEGY_HPP__
