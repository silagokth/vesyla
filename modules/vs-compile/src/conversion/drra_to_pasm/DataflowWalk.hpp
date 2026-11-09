#ifndef __VESYLA_CONVERSION_DRRA_TO_PASM_DATAFLOW_WALK_HPP__
#define __VESYLA_CONVERSION_DRRA_TO_PASM_DATAFLOW_WALK_HPP__

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace vesyla::conversion::drra_to_pasm {

// An evt rop the data flow from another evt reaches, and the cycles after it.
struct ReachedEvent {
  mlir::Operation *event;
  int32_t delay;
  // Loops the flow left through their result whose trip count changes from
  // pass to pass (AGU Slope Levels): the delay does not cover them, the
  // source's instance is the last one of such a loop instead.
  llvm::SmallVector<mlir::affine::AffineForOp, 2> last_of = {};
};

// Walks the data flow forward from the evt rop `source` through rops without
// an evt and loop iter_args, adding latencies, up to the next evt rops. These
// are the exact delays CreateConstraintsPass turns into `==` constraints.
llvm::SmallVector<ReachedEvent> reached_events(mlir::Operation *source);

// An evt rop whose data flow reaches `sink`, and the cycles it takes: the
// latencies of the rops without an evt in between. The backward twin of
// reached_events, within one loop body: operands with no defining rop, such as
// iter_args, end the walk.
llvm::SmallVector<ReachedEvent> feeding_events(mlir::Operation *sink);

// Evt rops under `scope` whose times those delays fix to be equal: each group
// has at least two members, which must run in the same cycle.
llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>
tied_groups(mlir::Operation *scope);

} // namespace vesyla::conversion::drra_to_pasm

#endif // __VESYLA_CONVERSION_DRRA_TO_PASM_DATAFLOW_WALK_HPP__
