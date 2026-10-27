#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <string>

#include "AssignIdsPass.hpp"

namespace vesyla::transformation::assign_ids {
#define GEN_PASS_DEF_ASSIGNIDSPASS
#include "transformation/assign_ids/Passes.hpp.inc"

namespace {

// The ops the passes below look up by id: buffers, accesses, and every arith
// op, since any of them can be the root a pattern is matched on.
bool needs_id(mlir::Operation *op) {
  return llvm::isa<mlir::memref::AllocOp, mlir::affine::AffineLoadOp,
                   mlir::affine::AffineStoreOp,
                   mlir::affine::AffineVectorLoadOp,
                   mlir::affine::AffineVectorStoreOp>(op) ||
         llvm::isa<mlir::arith::ArithDialect>(op->getDialect());
}

class AssignIdsPass : public impl::AssignIdsPassBase<AssignIdsPass> {
public:
  using impl::AssignIdsPassBase<AssignIdsPass>::AssignIdsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = &getContext();

    llvm::StringSet<> taken;
    module.walk([&](mlir::Operation *op) {
      if (auto id = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id"))
        taken.insert(id.getValue());
    });

    // Numbered per op name in program order, so the names read like the
    // program: @load_0, @load_1, @muli_0, ...
    llvm::StringMap<unsigned> next;
    module.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      if (!needs_id(op) || op->hasAttr("id"))
        return;
      std::string base = op->getName().stripDialect().str();
      std::string name;
      do {
        name = base + "_" + std::to_string(next[base]++);
      } while (taken.contains(name));
      taken.insert(name);
      op->setAttr("id", mlir::FlatSymbolRefAttr::get(context, name));
    });
  }
};

} // namespace
} // namespace vesyla::transformation::assign_ids
