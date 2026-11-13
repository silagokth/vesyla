#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "InterchangeLoopsPass.hpp"
#include "transformation/stage_io/IoAccess.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

namespace vesyla::transformation::interchange_loops {
#define GEN_PASS_DEF_INTERCHANGELOOPSPASS
#include "transformation/interchange_loops/Passes.hpp.inc"

namespace {

using mlir::affine::AffineForOp;
using stage_io::IoAccess;

// Position of each loop of `band` in `order`.
unsigned position(llvm::ArrayRef<AffineForOp> order, AffineForOp loop) {
  return llvm::find(order, loop) - order.begin();
}

// Whether a write's row stays fixed inside its word loops under `order`
// (outermost first): every loop its word reads but its row does not is
// inside every loop its row reads.
bool stageable(const IoAccess &write, llvm::ArrayRef<AffineForOp> order) {
  auto rows = stage_io::loops_of(write, write.row());
  for (AffineForOp word : stage_io::loops_of(write, write.word())) {
    if (llvm::is_contained(rows, word) || !llvm::is_contained(order, word))
      continue;
    for (AffineForOp row : rows)
      if (llvm::is_contained(order, row) &&
          position(order, word) < position(order, row))
        return false;
  }
  return true;
}

class InterchangeLoopsPass
    : public impl::InterchangeLoopsPassBase<InterchangeLoopsPass> {
public:
  using impl::InterchangeLoopsPassBase<
      InterchangeLoopsPass>::InterchangeLoopsPassBase;

  void runOnOperation() final {
    for (auto epoch : getOperation().getOps<pasm::EpochOp>()) {
      for (auto nest : llvm::to_vector(
               epoch.getBody().front().getOps<AffineForOp>())) {
        reorder(nest);
      }
    }
  }

private:
  void reorder(AffineForOp nest) {
    llvm::SmallVector<AffineForOp> band;
    mlir::affine::getPerfectlyNestedLoops(band, nest);
    if (band.size() < 2)
      return;

    llvm::SmallVector<IoAccess> writes;
    for (const IoAccess &access : stage_io::collect_io_accesses(nest))
      if (access.write)
        writes.push_back(access);
    if (llvm::all_of(writes, [&](const IoAccess &w) {
          return stageable(w, band);
        }))
      return;
    // Only accesses in the innermost body: a perfect nest.
    for (const IoAccess &w : writes)
      if (w.op->getParentOp() != band.back().getOperation())
        return;

    // Pure word loops of the failing writes go innermost, order kept.
    llvm::SmallVector<AffineForOp> sink;
    for (const IoAccess &w : writes) {
      if (stageable(w, band))
        continue;
      auto rows = stage_io::loops_of(w, w.row());
      for (AffineForOp word : stage_io::loops_of(w, w.word()))
        if (!llvm::is_contained(rows, word) && !llvm::is_contained(sink, word))
          sink.push_back(word);
    }
    llvm::SmallVector<AffineForOp> order;
    for (AffineForOp loop : band)
      if (!llvm::is_contained(sink, loop))
        order.push_back(loop);
    for (AffineForOp loop : band)
      if (llvm::is_contained(sink, loop))
        order.push_back(loop);
    if (!llvm::all_of(writes,
                      [&](const IoAccess &w) { return stageable(w, order); }))
      return;

    llvm::SmallVector<unsigned> perm;
    for (AffineForOp loop : band)
      perm.push_back(position(order, loop));
    if (!mlir::affine::isValidLoopInterchangePermutation(band, perm))
      return;
    mlir::affine::permuteLoops(band, perm);
  }
};

} // namespace
} // namespace vesyla::transformation::interchange_loops
