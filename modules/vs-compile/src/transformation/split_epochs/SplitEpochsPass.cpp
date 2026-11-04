#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "SplitEpochsPass.hpp"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

#include <map>
#include <set>
#include <string>
#include <tuple>

namespace vesyla::transformation::split_epochs {
#define GEN_PASS_DEF_SPLITEPOCHSPASS
#include "transformation/split_epochs/Passes.hpp.inc"

namespace {

// Config registers of a DPU.
constexpr size_t kDpuConfigs = 4;

using DpuKey = std::tuple<int32_t, int32_t, int32_t>;
using DpuConfigs = std::map<DpuKey, std::set<const void *>>;

// The distinct DPU configurations `op` runs, by DPU.
void collect(mlir::Operation *op, DpuConfigs &out) {
  op->walk([&](drra::RopOp rop) {
    auto kind = rop->getAttrOfType<mlir::StringAttr>("kind");
    if (!kind || kind.getValue() != "dpu" || !rop.getConfAttr()) {
      return;
    }
    pasm::ResourceAttr res;
    mlir::Attribute attr = rop->getAttr("resource");
    if (auto arr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
      res = arr.empty() ? nullptr : mlir::dyn_cast<pasm::ResourceAttr>(arr[0]);
    } else {
      res = mlir::dyn_cast_or_null<pasm::ResourceAttr>(attr);
    }
    if (!res) {
      return;
    }
    mlir::NamedAttrList conf(rop.getConfAttr());
    conf.erase("option");
    out[{res.getRow(), res.getCol(), res.getSlot()}].insert(
        conf.getDictionary(op->getContext()).getAsOpaquePointer());
  });
}

bool fits(const DpuConfigs &have, const DpuConfigs &add) {
  for (const auto &[dpu, configs] : add) {
    std::set<const void *> all = configs;
    auto it = have.find(dpu);
    if (it != have.end()) {
      all.insert(it->second.begin(), it->second.end());
    }
    if (all.size() > kDpuConfigs) {
      return false;
    }
  }
  return true;
}

void merge(DpuConfigs &have, const DpuConfigs &add) {
  for (const auto &[dpu, configs] : add) {
    have[dpu].insert(configs.begin(), configs.end());
  }
}

// True if an op from ops[cut] on uses a value defined before ops[cut].
bool crosses(llvm::ArrayRef<mlir::Operation *> ops, size_t cut,
             mlir::Block &top) {
  std::set<mlir::Operation *> before(ops.begin(), ops.begin() + cut);
  for (size_t i = cut; i < ops.size(); ++i) {
    bool uses = false;
    ops[i]->walk([&](mlir::Operation *op) {
      for (mlir::Value v : op->getOperands()) {
        mlir::Operation *def = v.getDefiningOp();
        if (def && before.count(top.findAncestorOpInBlock(*def))) {
          uses = true;
        }
      }
    });
    if (uses) {
      return true;
    }
  }
  return false;
}

class SplitEpochsPass : public impl::SplitEpochsPassBase<SplitEpochsPass> {
public:
  using impl::SplitEpochsPassBase<SplitEpochsPass>::SplitEpochsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::StringSet<> ids;
    llvm::SmallVector<pasm::EpochOp> work;
    for (auto epoch : module.getOps<pasm::EpochOp>()) {
      ids.insert(epoch.getId());
      work.push_back(epoch);
    }
    // Epochs are handled in program order; a new one goes next.
    for (size_t w = 0; w < work.size(); ++w) {
      if (auto next = split(work[w], ids)) {
        work.insert(work.begin() + w + 1, next);
      }
    }
  }

private:
  // Cuts `epoch` once if it has to; returns the new epoch, or null.
  pasm::EpochOp split(pasm::EpochOp epoch, llvm::StringSet<> &ids) {
    mlir::Block &top = epoch.getBody().front();
    llvm::SmallVector<mlir::Operation *> ops;
    for (mlir::Operation &op : top.without_terminator()) {
      ops.push_back(&op);
    }
    DpuConfigs have;
    for (size_t i = 0; i < ops.size(); ++i) {
      DpuConfigs add;
      collect(ops[i], add);
      if (i > 0 && !fits(have, add)) {
        for (size_t cut = i; cut > 0; --cut) {
          if (!crosses(ops, cut, top)) {
            return move_tail(epoch, ops, cut, ids);
          }
        }
      }
      merge(have, add);
    }
    return nullptr;
  }

  pasm::EpochOp move_tail(pasm::EpochOp epoch,
                          llvm::ArrayRef<mlir::Operation *> ops, size_t cut,
                          llvm::StringSet<> &ids) {
    std::string id;
    for (unsigned n = 1;; ++n) {
      id = epoch.getId().str() + "_s" + std::to_string(n);
      if (!ids.contains(id)) {
        break;
      }
    }
    ids.insert(id);
    mlir::OpBuilder builder(epoch.getContext());
    builder.setInsertionPointAfter(epoch);
    auto next = pasm::EpochOp::create(builder, epoch.getLoc(),
                                      builder.getStringAttr(id));
    builder.createBlock(&next.getBody());
    auto yield = pasm::YieldOp::create(builder, epoch.getLoc());
    for (size_t i = cut; i < ops.size(); ++i) {
      ops[i]->moveBefore(yield);
    }
    return next;
  }
};

} // namespace
} // namespace vesyla::transformation::split_epochs
