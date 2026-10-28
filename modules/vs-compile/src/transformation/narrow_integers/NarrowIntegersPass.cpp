#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "NarrowIntegersPass.hpp"

namespace vesyla::transformation::narrow_integers {
#define GEN_PASS_DEF_NARROWINTEGERSPASS
#include "transformation/narrow_integers/Passes.hpp.inc"

namespace {

// Rebuilds `wide` in `narrow`, or returns null when something under it cannot
// be. Each narrow op goes right before its wide counterpart, so it sees the
// narrow operands, which were built before theirs. `done` keeps a value shared
// by two truncations from being rebuilt twice; `replaced` collects the wide
// ops, operands before users.
mlir::Value narrow_value(mlir::Value wide, mlir::IntegerType narrow,
                         llvm::DenseMap<mlir::Value, mlir::Value> &done,
                         llvm::SmallVectorImpl<mlir::Operation *> &replaced) {
  if (mlir::Value known = done.lookup(wide)) {
    return known;
  }
  mlir::Operation *op = wide.getDefiningOp();
  if (!op) {
    return {};
  }
  mlir::OpBuilder builder(op);
  mlir::Value result;

  if (auto ext = llvm::dyn_cast<mlir::arith::ExtSIOp>(op)) {
    if (ext.getIn().getType() == narrow) {
      result = ext.getIn();
    }
  } else if (auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
    if (auto value = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())) {
      result = mlir::arith::ConstantOp::create(
          builder, op->getLoc(), narrow,
          builder.getIntegerAttr(narrow,
                                 value.getValue().trunc(narrow.getWidth())));
    }
  } else if (llvm::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                       mlir::arith::MulIOp>(op)) {
    mlir::Value lhs = narrow_value(op->getOperand(0), narrow, done, replaced);
    mlir::Value rhs =
        lhs ? narrow_value(op->getOperand(1), narrow, done, replaced)
            : mlir::Value();
    if (rhs) {
      // Built plain: an overflow flag the wide op carried does not hold in
      // the narrow type.
      mlir::Operation *rebuilt = nullptr;
      if (llvm::isa<mlir::arith::AddIOp>(op)) {
        rebuilt = mlir::arith::AddIOp::create(builder, op->getLoc(), lhs, rhs);
      } else if (llvm::isa<mlir::arith::SubIOp>(op)) {
        rebuilt = mlir::arith::SubIOp::create(builder, op->getLoc(), lhs, rhs);
      } else {
        rebuilt = mlir::arith::MulIOp::create(builder, op->getLoc(), lhs, rhs);
      }
      if (mlir::Attribute id = op->getAttr("id")) {
        rebuilt->setAttr("id", id);
      }
      result = rebuilt->getResult(0);
    }
  }

  if (result) {
    done[wide] = result;
    replaced.push_back(op);
  }
  return result;
}

class NarrowIntegersPass
    : public impl::NarrowIntegersPassBase<NarrowIntegersPass> {
public:
  using impl::NarrowIntegersPassBase<
      NarrowIntegersPass>::NarrowIntegersPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::arith::TruncIOp> truncs;
    module.walk([&](mlir::arith::TruncIOp trunc) { truncs.push_back(trunc); });

    llvm::DenseMap<mlir::Value, mlir::Value> done;
    llvm::SmallVector<mlir::Operation *> replaced;
    for (mlir::arith::TruncIOp trunc : truncs) {
      auto narrow = llvm::dyn_cast<mlir::IntegerType>(trunc.getType());
      if (!narrow) {
        continue;
      }
      if (mlir::Value narrowed =
              narrow_value(trunc.getIn(), narrow, done, replaced)) {
        trunc.getResult().replaceAllUsesWith(narrowed);
        trunc->erase();
      }
    }

    // The wide ops now feed nothing, unless some use outside a truncation
    // still needs them -- then they stay, and so does the extension under
    // them, which is reported below. Users come after their operands in
    // `replaced`, so walking it backwards frees each op before its operands.
    for (mlir::Operation *op : llvm::reverse(replaced)) {
      if (op->use_empty()) {
        op->erase();
      }
    }

    // Report rather than stop at the first: one run should show every op the
    // narrowing could not take.
    bool left = false;
    module.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                    mlir::arith::TruncIOp>(op)) {
        op->emitError("narrow-integers: this integer extension/truncation "
                      "could not be removed -- only add, sub and mul (and "
                      "constants) can be computed in the narrow type, and the "
                      "wide value may not be used outside a truncation");
        left = true;
      }
    });
    if (left) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::transformation::narrow_integers
