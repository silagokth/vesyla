#include "MemoryMap.hpp"
#include "Strategy.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmAttrs.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace vesyla {
namespace transformation {
namespace bind_resources {

namespace {

// The port the operation drives. Selection already fixed it: an rf word read is
// evt port 1, a bulk write evt port 2, a dpu rst evt port 1, and so on -- the
// port says which behaviour of the resource was selected, so it is not a
// binding decision. Binding picks (row, col, slot) and leaves the port alone.
int resolve_port(mlir::Operation *rop) {
  for (const char *name : {"evt", "conf"}) {
    auto dict = rop->getAttrOfType<mlir::DictionaryAttr>(name);
    if (!dict) {
      continue;
    }
    if (auto port = dict.getAs<mlir::IntegerAttr>("port")) {
      return static_cast<int>(port.getInt());
    }
  }
  return 0;
}

// The slot each value of an operation enters or leaves the resource by, given
// as an offset from the instance's first slot.
//
// A resource wider than one slot spreads its endpoints across them. The dpu is
// two slots, and its two narrow inputs are the target slots of two different
// switchbox channels -- the switchbox addresses a channel by slot and knows
// nothing of ports, so two operands that share a slot share a channel, and one
// of them never arrives. Which offset each operand and result uses is the
// resource's own business, so the library states it: `uses` cannot answer it,
// because those names are the resource author's and are deliberately opaque to
// the compiler, compared only against each other.
//
// The declaration is a dictionary of two integer arrays, in the operation's own
// result and operand order:
//
//   endpoints = {results = [0 : i32], operands = [0 : i32, 1 : i32]}
//
// Absent means every endpoint sits at the instance's first slot, which is what
// a single-slot resource wants and what all of them get.
struct EndpointSlots {
  llvm::SmallVector<int> results;
  llvm::SmallVector<int> operands;
};

// Read one of the two arrays, checking it against the count it has to match and
// against the slots the instance actually occupies.
mlir::LogicalResult read_offsets(mlir::Operation *rop,
                                 mlir::DictionaryAttr endpoints,
                                 llvm::StringRef which, unsigned expected,
                                 int size, llvm::SmallVectorImpl<int> &dst) {
  auto array = endpoints.getAs<mlir::ArrayAttr>(which);
  if (!array) {
    if (expected == 0) {
      return mlir::success();
    }
    mlir::InFlightDiagnostic diag =
        rop->emitError("bind-resources: `endpoints` declares no `");
    diag << which << "`, but the operation has " << expected << " of them";
    return mlir::failure();
  }
  if (array.size() != expected) {
    mlir::InFlightDiagnostic diag =
        rop->emitError("bind-resources: `endpoints` gives ");
    diag << array.size() << " " << which << " slot(s) for an operation with "
         << expected;
    return mlir::failure();
  }

  for (mlir::Attribute entry : array) {
    auto offset = mlir::dyn_cast<mlir::IntegerAttr>(entry);
    if (!offset) {
      rop->emitError("bind-resources: `endpoints` slot offsets must "
                     "be integers");
      return mlir::failure();
    }
    int value = static_cast<int>(offset.getInt());
    if (value < 0 || value >= size) {
      mlir::InFlightDiagnostic diag = rop->emitError(
          "bind-resources: `endpoints` names slot offset ");
      diag << value << ", but the instance is " << size << " slot(s) wide";
      return mlir::failure();
    }
    dst.push_back(value);
  }
  return mlir::success();
}

// What the operation declared. `slots` is left empty when it declared none,
// which every single-slot resource does.
mlir::LogicalResult read_endpoints(mlir::Operation *rop,
                                   const ResourceInstance &instance,
                                   std::optional<EndpointSlots> &slots) {
  auto endpoints = rop->getAttrOfType<mlir::DictionaryAttr>("endpoints");
  if (!endpoints) {
    // An operation with several endpoints on a resource that has a slot for
    // each of them has to say which goes where. Letting it through would give
    // every endpoint the instance's first slot, which reads as a working
    // binding and quietly routes two of them over one channel.
    if (instance.size > 1 && rop->getNumResults() + rop->getNumOperands() > 1) {
      mlir::InFlightDiagnostic diag =
          rop->emitError("bind-resources: this operation has ");
      diag << rop->getNumResults() + rop->getNumOperands()
           << " endpoints on a resource that is " << instance.size
           << " slots wide, but declares no `endpoints` -- the library has to "
              "say which slot each operand and result uses, or they all land "
              "on the first one";
      return mlir::failure();
    }
    slots.reset();
    return mlir::success();
  }

  EndpointSlots read;
  if (mlir::failed(read_offsets(rop, endpoints, "results", rop->getNumResults(),
                                instance.size, read.results)) ||
      mlir::failed(read_offsets(rop, endpoints, "operands",
                                rop->getNumOperands(), instance.size,
                                read.operands))) {
    return mlir::failure();
  }
  slots = std::move(read);
  return mlir::success();
}

// The binding decision, written the way everything downstream reads it: a
// single resource for an operation whose endpoints all sit on one slot, and the
// [results..., operands...] array for one that spreads them, which is the
// layout GenerateIcdepPass indexes.
mlir::Attribute build_resource(mlir::Operation *rop,
                               const ResourceInstance &instance,
                               const std::optional<EndpointSlots> &slots) {
  mlir::MLIRContext *ctx = rop->getContext();
  const int port = resolve_port(rop);
  auto at = [&](int offset) -> mlir::Attribute {
    return pasm::ResourceAttr::get(ctx, instance.row, instance.col,
                                   instance.slot + offset, port);
  };
  if (!slots) {
    return at(0);
  }

  llvm::SmallVector<mlir::Attribute> endpoints;
  for (int offset : slots->results) {
    endpoints.push_back(at(offset));
  }
  for (int offset : slots->operands) {
    endpoints.push_back(at(offset));
  }
  return mlir::ArrayAttr::get(ctx, endpoints);
}

// The memref each storage names: the op whose `id` is the storage symbol and
// whose one result is a memref (the module's memref.alloc).
llvm::DenseMap<mlir::Attribute, mlir::MemRefType>
storage_types(mlir::ModuleOp module) {
  llvm::DenseMap<mlir::Attribute, mlir::MemRefType> types;
  module.walk([&](mlir::Operation *op) {
    auto id = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id");
    if (!id || op->getNumResults() != 1) {
      return;
    }
    if (auto type = mlir::dyn_cast<mlir::MemRefType>(op->getResult(0).getType())) {
      types[id] = type;
    }
  });
  return types;
}

// The instruction an access issues, as (name, port), matched against what a
// memory takes (ResourceInstance::instructions).
llvm::SmallVector<std::pair<std::string, int>>
instructions_of(mlir::Operation *rop) {
  llvm::SmallVector<std::pair<std::string, int>> instructions;
  for (const char *name : {"conf", "evt"}) {
    auto dict = rop->getAttrOfType<mlir::DictionaryAttr>(name);
    if (!dict) {
      continue;
    }
    int port = -1;
    if (auto attr = dict.getAs<mlir::IntegerAttr>("port")) {
      port = static_cast<int>(attr.getInt());
    }
    instructions.push_back({name, port});
  }
  return instructions;
}

// Move an access's address to the storage's base row: the map's result goes up
// by the rows below it, counted in the access's own unit (a word, or a
// vector<N> for a bulk access). AffineToRepPass turns the map's constant into
// the AGU's init_addr.
mlir::LogicalResult move_map(mlir::Operation *rop, int base, int row_words) {
  if (base == 0) {
    return mlir::success();
  }
  mlir::MLIRContext *ctx = rop->getContext();
  mlir::Type value;
  if (rop->getNumResults() > 0) {
    value = rop->getResult(0).getType();
  } else if (rop->getNumOperands() > 0) {
    value = rop->getOperand(0).getType();
  }
  int64_t unit = 1;
  if (auto vector = mlir::dyn_cast_or_null<mlir::VectorType>(value)) {
    unit = vector.getNumElements();
  }
  if (unit <= 0 || row_words % unit != 0) {
    rop->emitError("bind-resources: an access of ")
        << unit << " words does not divide a memory row of " << row_words;
    return mlir::failure();
  }

  mlir::AffineMap map;
  if (auto attr = rop->getAttrOfType<mlir::AffineMapAttr>("map")) {
    map = attr.getValue();
  } else {
    // No map: address 0 on every iteration of the enclosing loops.
    unsigned loops = 0;
    for (mlir::Operation *parent = rop->getParentOp(); parent;
         parent = parent->getParentOp()) {
      loops += llvm::isa<mlir::affine::AffineForOp>(parent);
    }
    map = mlir::AffineMap::get(loops, 0, mlir::getAffineConstantExpr(0, ctx));
  }
  if (map.getNumResults() != 1) {
    rop->emitError("bind-resources: cannot move an address map with ")
        << map.getNumResults() << " results to a base row";
    return mlir::failure();
  }
  mlir::AffineExpr moved = map.getResult(0) + base * (row_words / unit);
  rop->setAttr("map", mlir::AffineMapAttr::get(mlir::AffineMap::get(
                          map.getNumDims(), map.getNumSymbols(), moved)));
  return mlir::success();
}

class GreedyColoringBinder : public Binder {
public:
  mlir::LogicalResult bind(const ConflictGraph &graph,
                           const Architecture &arch) final {
    if (graph.empty()) {
      return mlir::success();
    }

    // Most contended first. A node with many conflicts has the fewest instances
    // left to it, so deciding it while the pool is still open is what keeps the
    // colouring from needing more instances than the design has -- the ordering
    // costs a sort and regularly saves a colour.
    llvm::SmallVector<unsigned> order;
    for (unsigned node = 0; node < graph.size(); ++node) {
      order.push_back(node);
    }
    llvm::stable_sort(order, [&](unsigned a, unsigned b) {
      return graph.neighbors(a).size() > graph.neighbors(b).size();
    });

    // The architecture instance each node was given, indexed by node.
    llvm::SmallVector<int> assignment(graph.size(), -1);
    // Cache the per-kind candidate lists; a scope full of rf accesses should
    // not rescan the instance table for each one.
    llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<unsigned>> candidates;
    bool failed = mlir::failed(bind_memories(graph, arch, order, assignment));

    for (unsigned node : order) {
      if (graph.storage(node)) {
        continue;
      }
      // A node may be several operations -- every access to one register file
      // is bound as a unit -- so diagnostics are raised against the first of
      // them and the decision is written onto all of them.
      llvm::ArrayRef<mlir::Operation *> ops = graph.ops(node);
      mlir::Operation *rop = ops.front();
      mlir::StringAttr kind = graph.kind(node);
      if (!kind) {
        rop->emitError("bind-resources: operation carries no `kind`, "
                       "so there is nothing to bind it to -- instruction "
                       "selection should have written one");
        failed = true;
        continue;
      }

      auto cached = candidates.find(kind);
      if (cached == candidates.end()) {
        cached =
            candidates.insert({kind, arch.instances_of_kind(kind.getValue())})
                .first;
      }
      llvm::ArrayRef<unsigned> pool = cached->second;
      if (pool.empty()) {
        rop->emitError() << "bind-resources: the architecture "
                            "allocates no resource of kind '"
                         << kind.getValue() << "'";
        failed = true;
        continue;
      }


      // The instances the conflicting neighbours already hold are out; any
      // other instance of the kind will do.
      llvm::SmallDenseSet<unsigned> taken;
      for (unsigned neighbor : graph.neighbors(node)) {
        if (assignment[neighbor] >= 0) {
          taken.insert(static_cast<unsigned>(assignment[neighbor]));
        }
      }

      const unsigned *chosen = llvm::find_if(
          pool, [&](unsigned instance) { return !taken.contains(instance); });
      if (chosen == pool.end()) {
        // Every instance of the kind is held by something this operation
        // conflicts with. Spilling -- splitting the value out and putting it
        // back later -- is what would rescue this, and there is none yet, so
        // say what ran short instead of overcommitting an instance and letting
        // it surface as wrong results.
        rop->emitError()
            << "bind-resources: the architecture allocates "
            << pool.size() << " resource(s) of kind '" << kind.getValue()
            << "', but this conflicts with all of them -- the program needs "
               "more than the design has";
        failed = true;
        continue;
      }

      assignment[node] = static_cast<int>(*chosen);
      const ResourceInstance &instance = arch.at(*chosen);
      // The instance is the node's, but the port and the endpoint slots are
      // each operation's own: selection fixed the port when it chose the
      // behaviour, so a bulk write and a word read on one register file keep
      // theirs, and the library fixed which slot of the instance each operand
      // and result uses.
      for (mlir::Operation *bound : ops) {
        std::optional<EndpointSlots> slots;
        if (mlir::failed(read_endpoints(bound, instance, slots))) {
          failed = true;
          continue;
        }
        bound->setAttr("resource", build_resource(bound, instance, slots));
      }
    }

    return mlir::failure(failed);
  }

  void write_memory_map(llvm::raw_ostream &os) const final {
    if (map_ && points_) {
      map_->write(os, *points_);
    }
  }

private:
  std::optional<MemoryMap> map_;
  std::optional<ProgramPoints> points_;

  // The storage nodes, largest first and then in `order`: each gets the
  // smallest memory it fits in -- one holding nothing else over its lifetime
  // before one that does -- and the lowest rows free there.
  mlir::LogicalResult bind_memories(const ConflictGraph &graph,
                                    const Architecture &arch,
                                    llvm::ArrayRef<unsigned> order,
                                    llvm::SmallVectorImpl<int> &assignment) {
    llvm::SmallVector<unsigned> nodes;
    for (unsigned node : order) {
      if (graph.storage(node)) {
        nodes.push_back(node);
      }
    }
    if (nodes.empty()) {
      return mlir::success();
    }

    auto module =
        graph.ops(nodes.front()).front()->getParentOfType<mlir::ModuleOp>();
    mlir::FailureOr<ProgramPoints> points = ProgramPoints::build(module);
    if (mlir::failed(points)) {
      return mlir::failure();
    }
    points_ = std::move(*points);
    map_.emplace(arch);
    llvm::DenseMap<mlir::Attribute, mlir::MemRefType> types =
        storage_types(module);

    llvm::DenseMap<unsigned, int64_t> words;
    bool failed = false;
    for (unsigned node : nodes) {
      mlir::MemRefType type = types.lookup(graph.storage(node));
      if (!type || !type.hasStaticShape()) {
        graph.ops(node).front()->emitError()
            << "bind-resources: storage @"
            << graph.storage(node).getValue()
            << " names no memref of static shape, so its size is not known";
        failed = true;
        continue;
      }
      words[node] = type.getNumElements();
    }
    if (failed) {
      return mlir::failure();
    }
    llvm::stable_sort(nodes, [&](unsigned a, unsigned b) {
      return words[a] > words[b];
    });

    llvm::SmallVector<unsigned> memories = arch.memory_instances();
    for (unsigned node : nodes) {
      failed |= mlir::failed(
          bind_memory(graph, arch, memories, node, words[node], assignment));
    }
    return mlir::failure(failed);
  }

  mlir::LogicalResult bind_memory(const ConflictGraph &graph,
                                  const Architecture &arch,
                                  llvm::ArrayRef<unsigned> memories,
                                  unsigned node, int64_t words,
                                  llvm::SmallVectorImpl<int> &assignment) {
    llvm::ArrayRef<mlir::Operation *> ops = graph.ops(node);
    mlir::FlatSymbolRefAttr storage = graph.storage(node);
    Interval live = storage_lifetime(graph, node, words, *points_);

    // The memories that take every access to the storage.
    llvm::SmallVector<unsigned> taking;
    for (unsigned memory : memories) {
      const ResourceInstance &instance = arch.at(memory);
      bool takes = llvm::all_of(ops, [&](mlir::Operation *op) {
        return llvm::all_of(instructions_of(op), [&](const auto &instr) {
          return instance.has_instruction(instr.first, instr.second);
        });
      });
      if (takes && instance.row_words() > 0) {
        taking.push_back(memory);
      }
    }
    if (taking.empty()) {
      ops.front()->emitError()
          << "bind-resources: the architecture has no memory that "
             "takes the accesses to storage @"
          << storage.getValue();
      return mlir::failure();
    }

    // Big enough, smallest first (memory_instances is sorted by depth).
    llvm::SmallVector<unsigned> fitting;
    for (unsigned memory : taking) {
      if (arch.at(memory).depth() >= words) {
        fitting.push_back(memory);
      }
    }
    if (fitting.empty()) {
      const ResourceInstance &largest = arch.at(taking.back());
      ops.front()->emitError()
          << "bind-resources: storage @" << storage.getValue()
          << " needs " << words << " words, the largest memory in the "
          << "architecture holds " << largest.depth() << " (" << largest.kind
          << ")";
      return mlir::failure();
    }

    // Memories a conflicting storage already holds are out: the two would
    // want the same port at once.
    llvm::DenseMap<unsigned, unsigned> taken;
    for (unsigned neighbor : graph.neighbors(node)) {
      if (assignment[neighbor] >= 0) {
        taken[static_cast<unsigned>(assignment[neighbor])] = neighbor;
      }
    }

    auto rows_in = [&](const ResourceInstance &instance) {
      return static_cast<int>((words + instance.row_words() - 1) /
                              instance.row_words());
    };
    std::optional<std::pair<unsigned, int>> chosen;
    for (size_t group = 0; group < fitting.size() && !chosen;) {
      size_t end = group;
      int depth = arch.at(fitting[group]).depth();
      while (end < fitting.size() && arch.at(fitting[end]).depth() == depth) {
        ++end;
      }
      // Within one size, an idle memory first: two storages on one memory
      // take turns on it.
      for (bool want_busy : {false, true}) {
        for (size_t i = group; i < end && !chosen; ++i) {
          unsigned memory = fitting[i];
          if (taken.count(memory) || map_->busy(memory, live) != want_busy) {
            continue;
          }
          if (std::optional<int> base =
                  map_->fit(memory, rows_in(arch.at(memory)), live)) {
            chosen = {memory, *base};
          }
        }
      }
      group = end;
    }

    if (!chosen) {
      mlir::InFlightDiagnostic diag = ops.front()->emitError();
      diag << "bind-resources: storage @" << storage.getValue()
           << " (" << words << " words) fits in no memory over ["
           << points_->name(live.first) << ", " << points_->name(live.last)
           << "]:";
      for (unsigned memory : fitting) {
        const ResourceInstance &instance = arch.at(memory);
        diag << " " << instance.kind << " (" << instance.row << ","
             << instance.col << "," << instance.slot << ") ";
        if (auto conflict = taken.find(memory); conflict != taken.end()) {
          diag << "holds @" << graph.storage(conflict->second).getValue()
               << ", which conflicts with it;";
        } else {
          diag << map_->holders(memory, live, *points_) << ";";
        }
      }
      return mlir::failure();
    }

    auto [memory, base] = *chosen;
    const ResourceInstance &instance = arch.at(memory);
    map_->place(memory, base, rows_in(instance), {storage, live});
    assignment[node] = static_cast<int>(memory);

    mlir::Builder builder(ops.front()->getContext());
    bool failed = false;
    for (mlir::Operation *bound : ops) {
      std::optional<EndpointSlots> slots;
      if (mlir::failed(read_endpoints(bound, instance, slots)) ||
          mlir::failed(move_map(bound, base, instance.row_words()))) {
        failed = true;
        continue;
      }
      bound->setAttr("resource", build_resource(bound, instance, slots));
      bound->setAttr("kind", builder.getStringAttr(instance.kind));
      bound->setAttr("mem_base", builder.getI32IntegerAttr(base));
    }
    return mlir::failure(failed);
  }
};

} // namespace

std::unique_ptr<Binder> create_greedy_coloring_binder() {
  return std::make_unique<GreedyColoringBinder>();
}

} // namespace bind_resources
} // namespace transformation
} // namespace vesyla
