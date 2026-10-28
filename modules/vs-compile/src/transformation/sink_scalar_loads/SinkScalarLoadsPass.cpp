#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include "SinkScalarLoadsPass.hpp"
#include "conversion/select_instructions/NativeHelpers.hpp"

namespace vesyla::transformation::sink_scalar_loads {
#define GEN_PASS_DEF_SINKSCALARLOADSPASS
#include "transformation/sink_scalar_loads/Passes.hpp.inc"

namespace {

// The outermost loop around `user` that does not also hold `def`: the loop
// whose iterations all read the one value `def` computed. Null when `user`
// runs as often as `def` does. A loop's own operands (an iter_args init) are
// read before it starts, so the walk starts at the user's parent.
mlir::affine::AffineForOp outermost_crossed_loop(mlir::Operation *user,
                                                 mlir::Operation *def) {
  mlir::affine::AffineForOp crossed;
  for (mlir::Operation *parent = user->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->isAncestor(def)) {
      break;
    }
    if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(parent)) {
      crossed = loop;
    }
  }
  return crossed;
}

// Whether anything inside `loop` writes `memref`.
bool writes_inside(mlir::affine::AffineForOp loop, mlir::Value memref) {
  bool found = false;
  loop.walk([&](mlir::Operation *op) {
    if (auto write = llvm::dyn_cast<mlir::affine::AffineWriteOpInterface>(op)) {
      found |= write.getMemRef() == memref;
    } else if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
      found |= store.getMemRef() == memref;
    }
  });
  return found;
}

class SinkScalarLoadsPass
    : public impl::SinkScalarLoadsPassBase<SinkScalarLoadsPass> {
public:
  using impl::SinkScalarLoadsPassBase<
      SinkScalarLoadsPass>::SinkScalarLoadsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::affine::AffineLoadOp> loads;
    module.walk([&](mlir::affine::AffineLoadOp load) { loads.push_back(load); });

    bool failed = false;
    for (mlir::affine::AffineLoadOp load : loads) {
      // The uses inside a loop the load is outside of, by the block they sit
      // in, blocks in order of their first such use.
      llvm::MapVector<mlir::Block *, llvm::SmallVector<mlir::OpOperand *>>
          by_block;
      for (mlir::OpOperand &use : load->getUses()) {
        mlir::Operation *user = use.getOwner();
        mlir::affine::AffineForOp loop = outermost_crossed_loop(user, load);
        if (!loop) {
          continue;
        }
        if (vesyla::sel::is_io_buffer(load.getMemRef())) {
          user->emitError("sink-scalar-loads: this operation inside a loop "
                          "uses a word loaded from an io buffer outside it; "
                          "the io moves whole rows only (c-frontend G6)");
          failed = true;
          break;
        }
        if (writes_inside(loop, load.getMemRef())) {
          user->emitError("sink-scalar-loads: this operation inside a loop "
                          "uses a word loaded outside it from a buffer the "
                          "loop writes; re-reading it per iteration would "
                          "change the value (c-frontend G6)");
          failed = true;
          break;
        }
        by_block[user->getBlock()].push_back(&use);
      }

      // One copy per block, before its first use there; every use in the
      // block reads it. The copy has no id: AssignIdsPass names it.
      for (auto &[block, uses] : by_block) {
        mlir::Operation *first = uses.front()->getOwner();
        for (mlir::OpOperand *use : uses) {
          if (use->getOwner()->isBeforeInBlock(first)) {
            first = use->getOwner();
          }
        }
        mlir::OpBuilder builder(first);
        mlir::Operation *copy = builder.clone(*load);
        copy->removeAttr("id");
        for (mlir::OpOperand *use : uses) {
          use->set(copy->getResult(0));
        }
      }
      if (load->use_empty()) {
        load->erase();
      }
    }

    // Whatever else is computed once and used on every iteration has no
    // buffer to re-read it from. A constant is left alone: its stores are
    // select-instructions' to judge (c-frontend G10).
    module.walk([&](mlir::Operation *user) {
      for (mlir::Value value : user->getOperands()) {
        mlir::Operation *def = value.getDefiningOp();
        if (!def || !llvm::isa<mlir::IntegerType>(value.getType()) ||
            llvm::isa<mlir::arith::ConstantOp>(def) ||
            !outermost_crossed_loop(user, def)) {
          continue;
        }
        user->emitError("sink-scalar-loads: this operation inside a loop uses "
                        "a value computed outside it that is not a load; "
                        "not supported (c-frontend G6)");
        failed = true;
      }
    });

    if (failed) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::transformation::sink_scalar_loads
