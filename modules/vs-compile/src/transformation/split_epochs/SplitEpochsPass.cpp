#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "SplitEpochsPass.hpp"
#include "transformation/bind_resources/Architecture.hpp"
#include "vesyla/Support/AguMerge.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"
#include "vesyla/Dialect/Pasm/Transforms/ResourceFlow.hpp"
#include "vesyla/Dialect/Pasm/IR/OptionLimit.hpp"

#include <map>
#include <set>
#include <string>
#include <tuple>

namespace vesyla::transformation::split_epochs {
#define GEN_PASS_DEF_SPLITEPOCHSPASS
#include "transformation/split_epochs/Passes.hpp.inc"

namespace {

using ResourceKey = std::tuple<int32_t, int32_t, int32_t>;
using Configs = std::map<ResourceKey, std::set<const void *>>;
using Limits = std::map<ResourceKey, int64_t>;

// The distinct configurations (conf without its option) `op` runs, by the
// resource instance they are bound to.
void collect(mlir::Operation *op, Configs &out) {
  op->walk([&](drra::RopOp rop) {
    if (!rop.getConfAttr()) {
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

// NUM_OPTIONS of each resource instance (arch.json, 4 when not given), capped
// by what its instructions can name; 0 means its confs take no option and
// are not limited.
Limits read_limits(const bind_resources::Architecture &arch) {
  Limits limits;
  for (const bind_resources::ResourceInstance &r : arch.instances()) {
    limits[{r.row, r.col, r.slot}] =
        pasm::option_limit(r.kind, r.num_options());
  }
  return limits;
}

bool fits(const Configs &have, const Configs &add, const Limits &limits) {
  for (const auto &[res, configs] : add) {
    auto limit = limits.find(res);
    int64_t n = limit == limits.end() ? 4 : limit->second;
    if (n == 0) {
      continue;
    }
    std::set<const void *> all = configs;
    auto it = have.find(res);
    if (it != have.end()) {
      all.insert(it->second.begin(), it->second.end());
    }
    if (static_cast<int64_t>(all.size()) > n) {
      return false;
    }
  }
  return true;
}

void merge(Configs &have, const Configs &add) {
  for (const auto &[res, configs] : add) {
    have[res].insert(configs.begin(), configs.end());
  }
}

using PortKey = std::tuple<int32_t, int32_t, int32_t, int32_t>;
using Chains = llvm::SmallVector<llvm::SmallVector<const void *>>;
using PortUsers = std::map<PortKey, Chains>;

// The loop chains (enclosing affine.for, outermost first) of the evt rops in
// `op`, by the (row, col, slot, port) they are bound to.
void collect_ports(mlir::Operation *op, PortUsers &out) {
  op->walk([&](drra::RopOp rop) {
    auto res = mlir::dyn_cast_or_null<pasm::ResourceAttr>(rop->getAttr("resource"));
    if (!rop.getEvtAttr() || !res) {
      return;
    }
    llvm::SmallVector<const void *> chain;
    for (mlir::Operation *p = rop->getParentOp(); p; p = p->getParentOp()) {
      if (mlir::isa<mlir::affine::AffineForOp>(p)) {
        chain.insert(chain.begin(), p);
      }
    }
    out[{res.getRow(), res.getCol(), res.getSlot(), res.getPort()}].push_back(
        chain);
  });
}

// False if a port's rops, those before and those of `add`, could no longer
// form one AGU program (CoalesceRopsPass).
bool ports_merge(const PortUsers &have, const PortUsers &add) {
  for (const auto &[port, chains] : add) {
    auto it = have.find(port);
    if (it == have.end()) {
      continue;
    }
    Chains all = it->second;
    all.append(chains.begin(), chains.end());
    if (!::vesyla::util::loop_tree(all)) {
      return false;
    }
  }
  return true;
}

void merge_ports(PortUsers &have, const PortUsers &add) {
  for (const auto &[port, chains] : add) {
    have[port].append(chains.begin(), chains.end());
  }
}

// A word transfer the swb is configured for: source slot to target slot,
// in one cell.
using Link = std::tuple<int32_t, int32_t, int32_t, int32_t>; // row, col, src, dst
using SwbOption = std::set<Link>;
using SwbOptions = llvm::SmallVector<SwbOption>;

// The swb options the word transfers of `op` take, packed as the interconnect
// packs them: transfers in program order into one option, a new one when a
// target slot gets a second source (InterconnectBinding has_conflict). Sibling
// nests merge that way too; identical options are counted once by the caller.
SwbOptions collect_options(mlir::Operation *op) {
  SwbOptions options;
  op->walk<mlir::WalkOrder::PreOrder>([&](drra::RopOp rop) {
    for (mlir::OpResult result : rop->getResults()) {
      if (pasm::kind_of(result) != "word") {
        continue;
      }
      pasm::ResourceAttr src =
          pasm::get_result_resource(rop, result.getResultNumber());
      if (!src) {
        continue;
      }
      llvm::SmallVector<mlir::Attribute> dsts;
      llvm::SmallPtrSet<void *, 8> visited;
      pasm::collect_consumer_resources(result, dsts, visited);
      for (mlir::Attribute attr : dsts) {
        auto dst = mlir::dyn_cast<pasm::ResourceAttr>(attr);
        if (!dst) {
          continue;
        }
        Link link{src.getRow(), src.getCol(), src.getSlot(), dst.getSlot()};
        bool conflict = false;
        if (!options.empty()) {
          for (const Link &held : options.back()) {
            if (std::get<0>(held) == std::get<0>(link) &&
                std::get<1>(held) == std::get<1>(link) &&
                std::get<3>(held) == std::get<3>(link) &&
                std::get<2>(held) != std::get<2>(link)) {
              conflict = true;
            }
          }
        }
        if (options.empty() || conflict) {
          options.push_back({});
        }
        options.back().insert(link);
      }
    }
  });
  return options;
}

// The distinct options among `have` and `add`, per cell.
std::map<std::pair<int32_t, int32_t>, int64_t>
count_options(const SwbOptions &have, const SwbOptions &add) {
  std::map<std::pair<int32_t, int32_t>, std::set<SwbOption>> distinct;
  for (const SwbOptions *list : {&have, &add}) {
    for (const SwbOption &option : *list) {
      if (option.empty()) {
        continue;
      }
      const Link &first = *option.begin();
      distinct[{std::get<0>(first), std::get<1>(first)}].insert(option);
    }
  }
  std::map<std::pair<int32_t, int32_t>, int64_t> out;
  for (const auto &[cell, set] : distinct) {
    out[cell] = static_cast<int64_t>(set.size());
  }
  return out;
}

// False if the swb of some cell would need more options than it has.
bool options_fit(const SwbOptions &have, const SwbOptions &add,
                 const std::map<std::pair<int32_t, int32_t>, int64_t> &swb) {
  for (const auto &[cell, n] : count_options(have, add)) {
    auto it = swb.find(cell);
    if (it != swb.end() && it->second > 0 && n > it->second) {
      return false;
    }
  }
  return true;
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
    auto arch = bind_resources::Architecture::from_config(module);
    if (mlir::failed(arch)) {
      return signalPassFailure();
    }
    limits_ = read_limits(*arch);
    for (const bind_resources::ResourceInstance &r : arch->instances()) {
      if (r.kind == "swb") {
        swb_[{r.row, r.col}] = limits_[{r.row, r.col, r.slot}];
      }
    }
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
      if (failed_) {
        return signalPassFailure();
      }
    }
  }

private:
  Limits limits_;
  // Option limit of each cell's swb.
  std::map<std::pair<int32_t, int32_t>, int64_t> swb_;
  bool failed_ = false;

  // Cuts `epoch` once if it has to; returns the new epoch, or null.
  pasm::EpochOp split(pasm::EpochOp epoch, llvm::StringSet<> &ids) {
    mlir::Block &top = epoch.getBody().front();
    llvm::SmallVector<mlir::Operation *> ops;
    for (mlir::Operation &op : top.without_terminator()) {
      ops.push_back(&op);
    }
    Configs have;
    PortUsers have_ports;
    SwbOptions have_options;
    for (size_t i = 0; i < ops.size(); ++i) {
      Configs add;
      collect(ops[i], add);
      SwbOptions add_options = collect_options(ops[i]);
      if (!options_fit({}, add_options, swb_)) {
        std::map<std::pair<int32_t, int32_t>, int64_t> counts =
            count_options({}, add_options);
        for (const auto &[cell, n] : counts) {
          auto it = swb_.find(cell);
          if (it != swb_.end() && n > it->second) {
            ops[i]->emitError("split-epochs: the word transfers here need ")
                << n << " swb options, the architecture provides "
                << it->second << " (NUM_OPTIONS)";
          }
        }
        failed_ = true;
        return nullptr;
      }
      if (!fits({}, add, limits_)) {
        // One op cannot be cut further.
        for (const auto &[res, configs] : add) {
          auto limit = limits_.find(res);
          int64_t n = limit == limits_.end() ? 4 : limit->second;
          if (n > 0 && static_cast<int64_t>(configs.size()) > n) {
            ops[i]->emitError("split-epochs: slot ")
                << std::get<2>(res) << " needs " << configs.size()
                << " configurations here, the architecture provides " << n
                << " (NUM_OPTIONS)";
          }
        }
        failed_ = true;
        return nullptr;
      }
      PortUsers add_ports;
      collect_ports(ops[i], add_ports);
      if (i > 0 && (!fits(have, add, limits_) ||
                    !ports_merge(have_ports, add_ports) ||
                    !options_fit(have_options, add_options, swb_))) {
        for (size_t cut = i; cut > 0; --cut) {
          if (!crosses(ops, cut, top)) {
            return move_tail(epoch, ops, cut, ids);
          }
        }
      }
      merge(have, add);
      merge_ports(have_ports, add_ports);
      have_options.append(add_options.begin(), add_options.end());
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
