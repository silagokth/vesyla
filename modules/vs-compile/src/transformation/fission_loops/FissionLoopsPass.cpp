#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include "FissionLoopsPass.hpp"
#include "conversion/select_instructions/Matcher.hpp"
#include "conversion/select_instructions/PatternLibrary.hpp"
#include "conversion/select_instructions/Rewriter.hpp"
#include "transformation/bind_resources/Architecture.hpp"
#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

#include <cstdlib>
#include <string>

namespace vesyla::transformation::fission_loops {
#define GEN_PASS_DEF_FISSIONLOOPSPASS
#include "transformation/fission_loops/Passes.hpp.inc"

namespace {

namespace sel = vesyla::conversion::select_instructions;

llvm::StringRef kind_of(mlir::Operation *op) {
  if (!llvm::isa<drra::RopOp>(op))
    return {};
  auto kind = op->getAttrOfType<mlir::StringAttr>("kind");
  return kind ? kind.getValue() : llvm::StringRef();
}

bool is_dpu(mlir::Operation *op) { return kind_of(op) == "dpu"; }

// What BindResourcesPass gives a DPU of its own in a loop nest.
bool is_dpu_mode(mlir::Operation *op) {
  return is_dpu(op) && op->hasAttr("conf");
}

// DPUs needed at once: the modes around the inner loops plus the most one of
// them needs (sibling loops run one after the other and share).
unsigned count_dpu_modes(mlir::Operation *op) {
  if (is_dpu_mode(op))
    return 1;
  if (auto shared = op->getAttrOfType<mlir::IntegerAttr>("dpu_shared"))
    return shared.getInt();
  unsigned own = 0, inner = 0;
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region)
      for (mlir::Operation &nested : block) {
        unsigned n = count_dpu_modes(&nested);
        if (llvm::isa<mlir::affine::AffineForOp>(nested))
          inner = std::max(inner, n);
        else
          own += n;
      }
  return own + inner;
}

mlir::FlatSymbolRefAttr storage_of(mlir::Operation *op) {
  return op->getAttrOfType<mlir::FlatSymbolRefAttr>("storage");
}

// Ops every part may have its own copy of: memory reads and index arithmetic.
bool is_copyable(mlir::Operation *op) {
  if (llvm::isa<mlir::affine::AffineApplyOp, mlir::arith::ConstantOp>(op))
    return true;
  return llvm::isa<drra::RopOp>(op) && !is_dpu(op) && op->getNumResults() > 0 &&
         op->getNumOperands() == 0 && !op->hasAttr("conf");
}

// A row fetch: an RF write fed only by copyable ops (the io read).
bool is_fetch(mlir::Operation *op) {
  return llvm::isa<drra::RopOp>(op) && !is_dpu(op) &&
         op->getNumResults() == 0 && storage_of(op) &&
         llvm::all_of(op->getOperands(), [](mlir::Value v) {
           mlir::Operation *def = v.getDefiningOp();
           return def && is_copyable(def);
         });
}

// A fetch, or a loop of fetches and copyable ops.
bool only_fetches(mlir::Operation *op) {
  return !op->walk([](mlir::Operation *n) {
              if (is_fetch(n) || is_copyable(n) ||
                  llvm::isa<mlir::affine::AffineForOp,
                            mlir::affine::AffineYieldOp>(n))
                return mlir::WalkResult::advance();
              return mlir::WalkResult::interrupt();
            })
              .wasInterrupted();
}

bool reads_storage(mlir::Operation *op, mlir::FlatSymbolRefAttr storage) {
  bool found = false;
  op->walk([&](mlir::Operation *nested) {
    found |= llvm::isa<drra::RopOp>(nested) && nested->getNumResults() > 0 &&
             storage_of(nested) == storage;
  });
  return found;
}

// The op directly in `body` that holds `op`, or null.
mlir::Operation *ancestor_in(mlir::Block *body, mlir::Operation *op) {
  while (op && op->getBlock() != body)
    op = op->getParentOp();
  return op;
}

// The loops from the nest down to the innermost one of its perfect part.
llvm::SmallVector<mlir::affine::AffineForOp>
loop_chain(mlir::affine::AffineForOp nest) {
  llvm::SmallVector<mlir::affine::AffineForOp> chain{nest};
  while (true) {
    mlir::Block *body = chain.back().getBody();
    auto ops = body->without_terminator();
    if (std::next(ops.begin()) != ops.end() || ops.begin() == ops.end())
      break;
    auto inner = llvm::dyn_cast<mlir::affine::AffineForOp>(*ops.begin());
    if (!inner)
      break;
    chain.push_back(inner);
  }
  return chain;
}

class FissionLoopsPass : public impl::FissionLoopsPassBase<FissionLoopsPass> {
public:
  using impl::FissionLoopsPassBase<FissionLoopsPass>::FissionLoopsPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    auto arch = transformation::bind_resources::Architecture::from_config(module);
    if (mlir::failed(arch))
      return signalPassFailure();
    dpus_ = arch->instances_of_kind("dpu").size();
    for (unsigned i : arch->memory_instances())
      max_depth_ = std::max(max_depth_, arch->at(i).depth());
    if (dpus_ == 0)
      return;

    module.walk([&](mlir::Operation *op) {
      if (auto id = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id"))
        ids_.insert(id.getValue());
    });

    // Split one nest at a time; the copies it makes are checked again.
    for (unsigned guard = 0; guard < 1000; ++guard) {
      mlir::affine::AffineForOp over;
      for (auto epoch : module.getOps<pasm::EpochOp>()) {
        for (auto nest : epoch.getBody().front().getOps<mlir::affine::AffineForOp>()) {
          if (count_dpu_modes(nest) > dpus_) {
            over = nest;
            break;
          }
        }
        if (over)
          break;
      }
      if (!over)
        break;
      if (mlir::failed(split(module, over)))
        return signalPassFailure();
    }

    if (new_accesses_.empty())
      return;
    if (mlir::failed(select_new_accesses(module)))
      return signalPassFailure();
  }

private:
  unsigned dpus_ = 0;
  int max_depth_ = 0;
  llvm::StringSet<> ids_;
  unsigned buffers_ = 0;
  llvm::SmallVector<mlir::Operation *> new_accesses_;

  std::string fresh_id(llvm::StringRef base) {
    std::string name = base.str();
    for (unsigned n = 1; ids_.contains(name); ++n)
      name = base.str() + "_" + std::to_string(n);
    ids_.insert(name);
    return name;
  }

  using Deps =
      llvm::DenseMap<mlir::Operation *, llvm::SetVector<mlir::Operation *>>;

  // Body ops each one depends on through SSA values, and how many use each.
  static void dependences(mlir::Block *body,
                          llvm::ArrayRef<mlir::Operation *> ops, Deps &deps,
                          llvm::DenseMap<mlir::Operation *, unsigned> &users) {
    for (mlir::Operation *op : ops) {
      op->walk([&](mlir::Operation *nested) {
        for (mlir::Value v : nested->getOperands()) {
          mlir::Operation *def = v.getDefiningOp();
          if (!def)
            continue;
          mlir::Operation *at = ancestor_in(body, def);
          if (at && at != op)
            deps[op].insert(at);
        }
      });
    }
    for (auto &entry : deps)
      for (mlir::Operation *dep : entry.second)
        ++users[dep];
  }

  // Store trees packed into parts in program order while they fit; a tree
  // that alone needs too many DPUs is cut at its operations.
  mlir::LogicalResult
  pack(llvm::ArrayRef<mlir::Operation *> ops, Deps &deps,
       const llvm::DenseMap<mlir::Operation *, unsigned> &users,
       const llvm::DenseMap<mlir::Operation *, unsigned> &order,
       llvm::DenseMap<mlir::Operation *, unsigned> &part, unsigned &parts) {
    unsigned current = 0, used = 0;
    for (mlir::Operation *sink : ops) {
      if (is_copyable(sink) || users.lookup(sink))
        continue;
      llvm::SetVector<mlir::Operation *> tree;
      llvm::SmallVector<mlir::Operation *> work{sink};
      while (!work.empty()) {
        mlir::Operation *op = work.pop_back_val();
        if (is_copyable(op) || part.count(op) || !tree.insert(op))
          continue;
        for (mlir::Operation *dep : deps[op])
          work.push_back(dep);
      }
      auto members = tree.takeVector();
      llvm::sort(members, [&](mlir::Operation *a, mlir::Operation *b) {
        return order.lookup(a) < order.lookup(b);
      });
      unsigned need = 0;
      for (mlir::Operation *op : members)
        need += count_dpu_modes(op);
      if (used + need > dpus_ && need <= dpus_ && used > 0) {
        ++current;
        used = 0;
      }
      for (mlir::Operation *op : members) {
        unsigned n = count_dpu_modes(op);
        if (n > dpus_)
          return op->emitError("fission-loops: this operation alone needs ")
                 << n << " DPUs, the fabric has " << dpus_;
        if (used + n > dpus_) {
          ++current;
          used = 0;
        }
        part[op] = current;
        used += n;
      }
    }
    parts = current + 1;
    return mlir::success();
  }

  // A loop inside a nest that needs more DPUs than the fabric has: its body
  // is packed into groups that take turns on the DPUs every pass (the DPU
  // switches config registers between them). A value one group passes to a
  // later one goes through an RF word. Groups are marked `dpu_share` =
  // [loop, group], which binding lets share a DPU.
  mlir::LogicalResult share(mlir::ModuleOp module,
                            mlir::affine::AffineForOp loop) {
    auto chain = loop_chain(loop);
    mlir::Block *body = chain.back().getBody();
    llvm::SmallVector<mlir::Operation *> ops;
    for (mlir::Operation &op : body->without_terminator())
      ops.push_back(&op);
    for (mlir::Operation *op : ops)
      if (count_dpu_modes(op) > dpus_)
        if (auto inner = llvm::dyn_cast<mlir::affine::AffineForOp>(op))
          return share(module, inner);

    llvm::DenseMap<mlir::Operation *, unsigned> order;
    for (auto [i, op] : llvm::enumerate(ops))
      order[op] = i;
    Deps deps;
    llvm::DenseMap<mlir::Operation *, unsigned> users;
    dependences(body, ops, deps, users);
    llvm::DenseMap<mlir::Operation *, unsigned> part;
    unsigned parts;
    if (mlir::failed(pack(ops, deps, users, order, part, parts)))
      return mlir::failure();
    if (parts < 2)
      return loop.emitError("fission-loops: needs ")
             << count_dpu_modes(loop) << " DPUs but cannot be shared";

    mlir::MLIRContext *ctx = module.getContext();
    mlir::OpBuilder builder(ctx);
    auto tag = mlir::FlatSymbolRefAttr::get(ctx, fresh_id("dpu_share"));
    unsigned need = 0;
    llvm::DenseMap<unsigned, unsigned> per_group;
    for (mlir::Operation *op : ops) {
      if (!part.count(op))
        continue;
      auto group = mlir::ArrayAttr::get(
          ctx, {tag, builder.getI32IntegerAttr(part[op])});
      op->walk([&](mlir::Operation *nested) {
        if (is_dpu_mode(nested))
          nested->setAttr("dpu_share", group);
      });
      need = std::max(need, per_group[part[op]] += count_dpu_modes(op));
    }

    // Values crossing to a later group: one RF word, written after the
    // producer, read before the first user of each later group.
    for (mlir::Operation *op : ops) {
      if (is_copyable(op) || !part.count(op))
        continue;
      for (mlir::Value result : op->getResults()) {
        llvm::MapVector<unsigned, mlir::Operation *> first;
        for (mlir::Operation *user : result.getUsers()) {
          mlir::Operation *at = ancestor_in(body, user);
          if (!at)
            return user->emitError("fission-loops: uses a value of a shared "
                                   "loop outside of it");
          // A yielded value is the loop's carry, kept by its own group.
          if (at == body->getTerminator())
            continue;
          unsigned k = part.lookup(at);
          if (k == part[op])
            continue;
          if (k < part[op])
            return user->emitError("fission-loops: internal error, a group "
                                   "uses a value of a later one");
          auto [it, fresh] = first.insert({k, at});
          if (!fresh && order[at] < order[it->second])
            it->second = at;
        }
        if (first.empty())
          continue;
        builder.setInsertionPoint(
            (*module.getOps<pasm::EpochOp>().begin()).getOperation());
        auto alloc = mlir::memref::AllocOp::create(
            builder, op->getLoc(), mlir::MemRefType::get({1}, result.getType()));
        std::string name = fresh_id("share_" + std::to_string(buffers_++));
        alloc->setAttr("id", mlir::FlatSymbolRefAttr::get(ctx, name));
        auto zero = mlir::AffineMap::get(0, 0, mlir::getAffineConstantExpr(0, ctx));
        builder.setInsertionPointAfter(op);
        auto store = mlir::affine::AffineStoreOp::create(
            builder, op->getLoc(), result, alloc, zero, mlir::ValueRange{});
        store->setAttr("id", mlir::FlatSymbolRefAttr::get(ctx, fresh_id(name + "_w")));
        new_accesses_.push_back(store);
        for (auto [k, at] : first) {
          builder.setInsertionPoint(at);
          auto load = mlir::affine::AffineLoadOp::create(
              builder, op->getLoc(), alloc, zero, mlir::ValueRange{});
          load->setAttr("id", mlir::FlatSymbolRefAttr::get(ctx, fresh_id(name + "_r")));
          new_accesses_.push_back(load);
          result.replaceUsesWithIf(load, [&, k = k](mlir::OpOperand &use) {
            mlir::Operation *u = ancestor_in(body, use.getOwner());
            return u && part.lookup(u) == k && use.getOwner() != store;
          });
        }
      }
    }
    loop->setAttr("dpu_shared", builder.getI32IntegerAttr(need));
    return mlir::success();
  }

  mlir::LogicalResult split(mlir::ModuleOp module,
                            mlir::affine::AffineForOp nest) {
    auto epoch = llvm::cast<pasm::EpochOp>(nest->getParentOp());
    auto chain = loop_chain(nest);
    mlir::Block *body = chain.back().getBody();
    llvm::SmallVector<mlir::Operation *> ops;
    for (mlir::Operation &op : body->without_terminator())
      ops.push_back(&op);
    llvm::DenseMap<mlir::Operation *, unsigned> order;
    for (auto [i, op] : llvm::enumerate(ops))
      order[op] = i;

    Deps deps;
    llvm::DenseMap<mlir::Operation *, unsigned> users;
    dependences(body, ops, deps, users);

    // A loop in the body that alone needs too many DPUs shares them instead.
    for (mlir::Operation *op : ops)
      if (count_dpu_modes(op) > dpus_)
        if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(op))
          return share(module, loop);

    llvm::DenseMap<mlir::Operation *, unsigned> part;
    unsigned parts;
    if (mlir::failed(pack(ops, deps, users, order, part, parts)))
      return mlir::failure();
    if (parts < 2)
      return nest.emitError("fission-loops: needs ")
             << count_dpu_modes(nest) << " DPUs but cannot be split";

    // Fetches are repeated in every part that reads what they wrote.
    llvm::DenseMap<mlir::Operation *, llvm::SmallSetVector<unsigned, 4>> in;
    for (mlir::Operation *op : ops)
      if (!is_copyable(op))
        in[op].insert(part[op]);
    for (mlir::Operation *op : ops) {
      if (is_copyable(op))
        continue;
      llvm::SmallSetVector<mlir::Attribute, 4> written;
      op->walk([&](mlir::Operation *nested) {
        if (nested->getNumResults() == 0)
          if (auto storage = storage_of(nested))
            written.insert(storage);
      });
      for (mlir::Attribute attr : written)
        for (mlir::Operation *reader : ops) {
          auto storage = llvm::cast<mlir::FlatSymbolRefAttr>(attr);
          if (is_copyable(reader) || reader == op ||
              !reads_storage(reader, storage) || part[reader] == part[op])
            continue;
          if (!only_fetches(op))
            return reader->emitError("fission-loops: reads @")
                   << storage.getValue()
                   << " in another part than the loop writes it";
          in[op].insert(part[reader]);
        }
    }

    // A carried value belongs to the part that computes it; the other copies
    // of the loop do without it.
    auto inner = chain.back();
    unsigned carries = inner.getNumIterOperands();
    llvm::SmallVector<unsigned> owner(carries, 0);
    if (carries) {
      if (inner != nest)
        return inner.emitError("fission-loops: a split loop inside a perfect "
                               "nest carries a value; not supported");
      mlir::Operation *yield = body->getTerminator();
      for (unsigned i = 0; i < carries; ++i) {
        mlir::Operation *def = yield->getOperand(i).getDefiningOp();
        mlir::Operation *at = def ? ancestor_in(body, def) : nullptr;
        if (at && part.count(at))
          owner[i] = part[at];
      }
    }

    // Values that cross from one part to a later one.
    struct Crossing {
      mlir::Value value;
      llvm::SmallVector<unsigned> to;
    };
    llvm::SmallVector<Crossing> crossings;
    for (mlir::Operation *op : ops) {
      if (is_copyable(op))
        continue;
      for (mlir::Value result : op->getResults()) {
        llvm::SmallSetVector<unsigned, 4> to;
        for (mlir::Operation *user : result.getUsers()) {
          mlir::Operation *at = ancestor_in(body, user);
          if (!at)
            return user->emitError("fission-loops: uses a value of a split "
                                   "loop outside of it");
          if (at == body->getTerminator())
            continue;
          if (part[at] < part[op])
            return user->emitError("fission-loops: internal error, a part uses "
                                   "a value of a later one");
          if (part[at] != part[op])
            to.insert(part[at]);
        }
        if (!to.empty())
          crossings.push_back({result, llvm::to_vector(to)});
      }
    }

    // A crossing value gets one word per iteration of the loop chain.
    int64_t words = 1;
    for (auto loop : chain) {
      if (!loop.hasConstantBounds() || loop.getConstantLowerBound() != 0 ||
          loop.getStepAsInt() != 1)
        return loop.emitError("fission-loops: a split loop needs constant "
                              "bounds from 0 with step 1");
      words *= loop.getConstantUpperBound();
    }
    if (!crossings.empty() && max_depth_ > 0 && words > max_depth_)
      return nest.emitError("fission-loops: a value crossing parts needs ")
             << words << " words, the largest register file holds "
             << max_depth_;

    // Copies of the nest, one per part, one after the other in the epoch.
    mlir::MLIRContext *ctx = module.getContext();
    mlir::OpBuilder builder(ctx);
    llvm::SmallVector<mlir::Operation *> nests{nest};
    llvm::SmallVector<mlir::IRMapping> maps(parts);
    for (mlir::Operation *op : ops)
      maps[0].map(op, op);
    for (unsigned k = 1; k < parts; ++k) {
      builder.setInsertionPointAfter(nests.back());
      mlir::Operation *copy = builder.clone(*nest, maps[k]);
      copy->walk([&](mlir::Operation *op) {
        if (auto id = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id"))
          op->setAttr("id", mlir::FlatSymbolRefAttr::get(
                                ctx, fresh_id(id.getValue().str() + "_p" +
                                              std::to_string(k))));
      });
      nests.push_back(copy);
    }

    // The word a crossing value goes to: the loop chain's counters, linearised.
    auto address = [&](unsigned k) {
      llvm::SmallVector<mlir::Value> ivs;
      mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, ctx);
      int64_t stride = 1;
      for (int i = chain.size() - 1; i >= 0; --i) {
        expr = expr + mlir::getAffineDimExpr(i, ctx) * stride;
        stride *= chain[i].getConstantUpperBound();
      }
      for (auto loop : chain)
        ivs.push_back(maps[k].lookupOrDefault(loop.getInductionVar()));
      return std::make_pair(mlir::AffineMap::get(chain.size(), 0, expr), ivs);
    };

    for (Crossing &crossing : crossings) {
      mlir::Operation *def = crossing.value.getDefiningOp();
      auto type = mlir::MemRefType::get({words}, crossing.value.getType());
      builder.setInsertionPoint(
          (*module.getOps<pasm::EpochOp>().begin()).getOperation());
      auto alloc = mlir::memref::AllocOp::create(builder, def->getLoc(), type);
      std::string name = fresh_id("split_" + std::to_string(buffers_++));
      alloc->setAttr("id", mlir::FlatSymbolRefAttr::get(ctx, name));

      unsigned from = part[ancestor_in(body, def)];
      mlir::Operation *producer = maps[from].lookupOrDefault(
          ancestor_in(body, def));
      builder.setInsertionPointAfter(producer);
      auto [map, ivs] = address(from);
      auto store = mlir::affine::AffineStoreOp::create(
          builder, def->getLoc(), maps[from].lookupOrDefault(crossing.value),
          alloc, map, ivs);
      store->setAttr("id", mlir::FlatSymbolRefAttr::get(ctx, fresh_id(name + "_w")));
      new_accesses_.push_back(store);

      for (unsigned k : crossing.to) {
        mlir::Operation *first = nullptr;
        for (mlir::Operation *user : crossing.value.getUsers()) {
          mlir::Operation *at = ancestor_in(body, user);
          if (part[at] == k && (!first || order[at] < order[first]))
            first = at;
        }
        builder.setInsertionPoint(maps[k].lookupOrDefault(first));
        auto [map, ivs] = address(k);
        auto load = mlir::affine::AffineLoadOp::create(builder, def->getLoc(),
                                                       alloc, map, ivs);
        load->setAttr("id", mlir::FlatSymbolRefAttr::get(
                                ctx, fresh_id(name + "_r")));
        new_accesses_.push_back(load);
        maps[k].lookupOrDefault(crossing.value).replaceAllUsesWith(load);
      }
    }

    // Outside, the loop's carried results come from the copy that keeps them;
    // the others yield their carry unchanged until rebuilt without it below.
    for (unsigned i = 0; i < carries; ++i) {
      auto keeper = llvm::cast<mlir::affine::AffineForOp>(nests[owner[i]]);
      if (owner[i] != 0)
        nest->getResult(i).replaceAllUsesWith(keeper.getResult(i));
      for (unsigned k = 0; k < parts; ++k) {
        if (k == owner[i])
          continue;
        auto copy = llvm::cast<mlir::affine::AffineForOp>(nests[k]);
        copy.getBody()->getTerminator()->setOperand(
            i, copy.getRegionIterArgs()[i]);
      }
    }

    // Each copy keeps its part and what it reads.
    for (unsigned k = 0; k < parts; ++k) {
      for (mlir::Operation *op : llvm::reverse(ops)) {
        if (is_copyable(op) || in[op].contains(k))
          continue;
        mlir::Operation *copy = maps[k].lookupOrDefault(op);
        for (mlir::Value result : copy->getResults())
          if (!result.use_empty())
            return op->emitError("fission-loops: internal error, a removed "
                                 "operation is still used");
        copy->erase();
      }
      mlir::Block *copy_body =
          llvm::cast<mlir::affine::AffineForOp>(
              maps[k].lookupOrDefault(chain.back().getOperation()))
              .getBody();
      for (bool changed = true; changed;) {
        changed = false;
        for (mlir::Operation &op :
             llvm::make_early_inc_range(llvm::reverse(*copy_body))) {
          if (is_copyable(&op) && op.use_empty()) {
            op.erase();
            changed = true;
          }
        }
      }
    }

    // Copies that keep no carry are rebuilt without it.
    for (unsigned k = 0; k < parts && carries; ++k) {
      auto copy = llvm::cast<mlir::affine::AffineForOp>(nests[k]);
      llvm::SmallVector<unsigned> kept;
      for (unsigned i = 0; i < carries; ++i)
        if (owner[i] == k)
          kept.push_back(i);
      if (kept.size() == carries)
        continue;
      llvm::SmallVector<mlir::Value> inits;
      for (unsigned i : kept)
        inits.push_back(copy.getInits()[i]);
      builder.setInsertionPoint(copy);
      auto rebuilt = mlir::affine::AffineForOp::create(
          builder, copy.getLoc(), copy.getConstantLowerBound(),
          copy.getConstantUpperBound(), copy.getStepAsInt(), inits);
      rebuilt->setDiscardableAttrs(copy->getDiscardableAttrDictionary());
      mlir::Block *from = copy.getBody();
      mlir::Block *to = rebuilt.getBody();
      copy.getInductionVar().replaceAllUsesWith(rebuilt.getInductionVar());
      for (auto [j, i] : llvm::enumerate(kept))
        copy.getRegionIterArgs()[i].replaceAllUsesWith(
            rebuilt.getRegionIterArgs()[j]);
      mlir::Operation *yield = from->getTerminator();
      llvm::SmallVector<mlir::Value> yielded;
      for (unsigned i : kept)
        yielded.push_back(yield->getOperand(i));
      if (!to->empty())
        to->back().erase(); // the yield a loop without carries is built with
      to->getOperations().splice(to->end(), from->getOperations(),
                                 from->begin(), mlir::Block::iterator(yield));
      builder.setInsertionPointToEnd(to);
      mlir::affine::AffineYieldOp::create(builder, yield->getLoc(), yielded);
      for (auto [j, i] : llvm::enumerate(kept))
        copy.getResult(i).replaceAllUsesWith(rebuilt.getResult(j));
      for (mlir::Value r : copy.getResults())
        if (!r.use_empty())
          return copy.emitError("fission-loops: internal error, a dropped "
                                "carry is still used");
      copy->erase();
      nests[k] = rebuilt;
    }

    // A carry's start value (the accumulator clear) goes with its copy, so it
    // does not hold the DPU across the copies before it.
    for (unsigned i = 0; i < carries; ++i) {
      auto keeper = llvm::cast<mlir::affine::AffineForOp>(nests[owner[i]]);
      unsigned j = llvm::count_if(llvm::seq(0u, i),
                                  [&](unsigned c) { return owner[c] == owner[i]; });
      mlir::Operation *init = keeper.getInits()[j].getDefiningOp();
      if (init && init->getBlock() == keeper->getBlock() &&
          init->isBeforeInBlock(nests.front()) && init->hasOneUse())
        init->moveBefore(keeper);
    }

    // Fetches before the nest go before the part of their first reader.
    mlir::Block &top = epoch.getBody().front();
    llvm::SmallVector<mlir::Operation *> before;
    for (mlir::Operation &op : top.without_terminator()) {
      if (&op == nests.front())
        break;
      before.push_back(&op);
    }
    for (mlir::Operation *op : before) {
      mlir::FlatSymbolRefAttr storage = storage_of(op);
      if (!is_fetch(op) || !storage)
        continue;
      bool single = llvm::all_of(op->getOperands(), [&](mlir::Value v) {
        return v.hasOneUse() && v.getDefiningOp()->getBlock() == &top;
      });
      if (!single)
        continue;
      unsigned k = 0;
      while (k < parts && !reads_storage(nests[k], storage))
        ++k;
      if (k == 0 || k == parts)
        continue;
      for (mlir::Value v : op->getOperands())
        v.getDefiningOp()->moveBefore(nests[k]);
      op->moveBefore(nests[k]);
    }
    return mlir::success();
  }

  // The new buffer accesses become rops the way instruction selection makes
  // them, from the RF patterns of the component library.
  mlir::LogicalResult select_new_accesses(mlir::ModuleOp module) {
    const char *env = std::getenv("VESYLA_SUITE_PATH_COMPONENTS");
    if (!env || !*env)
      return module.emitError(
          "fission-loops: no component path -- set VESYLA_SUITE_PATH_COMPONENTS");
    sel::PatternLibrary library;
    if (mlir::failed(library.load(env, module.getContext(), llvm::errs())))
      return mlir::failure();
    for (mlir::Operation *access : new_accesses_) {
      bool done = false;
      for (const auto &tier : library.tiers()) {
        for (const sel::Pattern *pattern : tier) {
          sel::MatchResult result;
          if (!sel::Matcher(*pattern).match(access, result) ||
              result.cone.size() != 1 || result.cone.front() != access)
            continue;
          if (mlir::failed(
                  sel::applyReplacement(*pattern, access, result, llvm::errs())))
            return mlir::failure();
          access->erase();
          done = true;
          break;
        }
        if (done)
          break;
      }
      if (!done)
        return access->emitError("fission-loops: no pattern selects this access");
    }
    return mlir::success();
  }
};

} // namespace
} // namespace vesyla::transformation::fission_loops
