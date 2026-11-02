#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "CopyStoragesPass.hpp"
#include "conversion/drra_to_pasm/DataflowWalk.hpp"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

#include <string>

namespace vesyla::transformation::copy_storages {
#define GEN_PASS_DEF_COPYSTORAGESPASS
#include "transformation/copy_storages/Passes.hpp.inc"

namespace {

mlir::FlatSymbolRefAttr storage_of(mlir::Operation *op) {
  return op->getAttrOfType<mlir::FlatSymbolRefAttr>("storage");
}

int64_t evt_port(mlir::Operation *op) {
  auto evt = op->getAttrOfType<mlir::DictionaryAttr>("evt");
  auto port = evt ? evt.getAs<mlir::IntegerAttr>("port") : nullptr;
  return port ? port.getInt() : -1;
}

// An RF write fed by io reads only.
bool is_fetch(mlir::Operation *op) {
  if (!llvm::isa<drra::RopOp>(op) || op->getNumResults() != 0 ||
      op->getNumOperands() == 0)
    return false;
  return llvm::all_of(op->getOperands(), [](mlir::Value v) {
    auto def = v.getDefiningOp<drra::RopOp>();
    auto kind = def ? def->getAttrOfType<mlir::StringAttr>("kind") : nullptr;
    return kind && kind.getValue() == "io" && v.hasOneUse();
  });
}

class CopyStoragesPass : public impl::CopyStoragesPassBase<CopyStoragesPass> {
public:
  using impl::CopyStoragesPassBase<CopyStoragesPass>::CopyStoragesPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::Operation *op) {
      if (auto id = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id"))
        ids_.insert(id.getValue());
    });

    // Per storage, the copies made so far; reader n of a tied set uses copy n.
    llvm::DenseMap<mlir::Attribute, llvm::SmallVector<mlir::FlatSymbolRefAttr>>
        copies;
    for (auto epoch : llvm::to_vector(module.getOps<pasm::EpochOp>())) {
      for (auto &group : conversion::drra_to_pasm::tied_groups(epoch)) {
        llvm::MapVector<std::pair<mlir::Attribute, int64_t>,
                        llvm::SmallVector<mlir::Operation *>>
            same;
        for (mlir::Operation *op : group)
          if (auto storage = storage_of(op))
            same[{storage, evt_port(op)}].push_back(op);
        for (auto &entry : same) {
          auto &ops = entry.second;
          if (ops.size() < 2)
            continue;
          auto storage = llvm::cast<mlir::FlatSymbolRefAttr>(entry.first.first);
          if (ops.front()->getNumResults() == 0) {
            ops[1]->emitError("copy-storages: two writes to @")
                << storage.getValue() << " must happen in the same cycle";
            return signalPassFailure();
          }
          for (size_t n = 1; n < ops.size(); ++n) {
            auto &made = copies[storage];
            while (made.size() < n) {
              auto copy = make_copy(module, storage, made.size() + 1);
              if (mlir::failed(copy))
                return signalPassFailure();
              made.push_back(*copy);
            }
            ops[n]->setAttr("storage", made[n - 1]);
          }
        }
      }
    }
  }

private:
  llvm::StringSet<> ids_;

  std::string fresh_id(llvm::StringRef base) {
    std::string name = base.str();
    for (unsigned n = 1; ids_.contains(name); ++n)
      name = base.str() + "_" + std::to_string(n);
    ids_.insert(name);
    return name;
  }

  // A new storage like `storage`, filled by clones of every fetch filling it.
  mlir::FailureOr<mlir::FlatSymbolRefAttr>
  make_copy(mlir::ModuleOp module, mlir::FlatSymbolRefAttr storage,
            size_t n) {
    mlir::MLIRContext *ctx = module.getContext();
    mlir::memref::AllocOp alloc;
    module.walk([&](mlir::memref::AllocOp op) {
      if (op->getAttrOfType<mlir::FlatSymbolRefAttr>("id") == storage)
        alloc = op;
    });
    if (!alloc)
      return module.emitError("copy-storages: no memref.alloc for @")
             << storage.getValue(), mlir::failure();

    llvm::SmallVector<mlir::Operation *> writes;
    bool bad = false;
    module.walk([&](drra::RopOp rop) {
      if (storage_of(rop) != storage || rop->getNumResults() != 0)
        return;
      if (!is_fetch(rop)) {
        rop->emitError("copy-storages: @")
            << storage.getValue()
            << " is read twice in one cycle and written by something other "
               "than an io fetch, so it cannot be copied";
        bad = true;
      }
      writes.push_back(rop);
    });
    if (bad)
      return mlir::failure();

    mlir::OpBuilder builder(alloc);
    builder.setInsertionPointAfter(alloc);
    auto copy_alloc = llvm::cast<mlir::memref::AllocOp>(builder.clone(*alloc));
    auto name = mlir::FlatSymbolRefAttr::get(
        ctx, fresh_id(storage.getValue().str() + "_c" + std::to_string(n)));
    copy_alloc->setAttr("id", name);

    for (mlir::Operation *write : writes) {
      builder.setInsertionPointAfter(write);
      mlir::IRMapping map;
      for (mlir::Value v : write->getOperands()) {
        mlir::Operation *read = builder.clone(*v.getDefiningOp(), map);
        rename(read);
      }
      mlir::Operation *clone = builder.clone(*write, map);
      rename(clone);
      clone->setAttr("storage", name);
    }
    return name;
  }

  void rename(mlir::Operation *op) {
    if (auto id = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id"))
      op->setAttr("id", mlir::FlatSymbolRefAttr::get(
                            op->getContext(), fresh_id(id.getValue().str() + "_c")));
  }
};

} // namespace
} // namespace vesyla::transformation::copy_storages
