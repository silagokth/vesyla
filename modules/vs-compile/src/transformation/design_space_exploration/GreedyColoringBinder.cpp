#include "Strategy.hpp"

#include "mlir/IR/BuiltinAttributes.h"
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
namespace dse {

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
        rop->emitError("design-space-exploration: `endpoints` declares no `");
    diag << which << "`, but the operation has " << expected << " of them";
    return mlir::failure();
  }
  if (array.size() != expected) {
    mlir::InFlightDiagnostic diag =
        rop->emitError("design-space-exploration: `endpoints` gives ");
    diag << array.size() << " " << which << " slot(s) for an operation with "
         << expected;
    return mlir::failure();
  }

  for (mlir::Attribute entry : array) {
    auto offset = mlir::dyn_cast<mlir::IntegerAttr>(entry);
    if (!offset) {
      rop->emitError("design-space-exploration: `endpoints` slot offsets must "
                     "be integers");
      return mlir::failure();
    }
    int value = static_cast<int>(offset.getInt());
    if (value < 0 || value >= size) {
      mlir::InFlightDiagnostic diag = rop->emitError(
          "design-space-exploration: `endpoints` names slot offset ");
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
          rop->emitError("design-space-exploration: this operation has ");
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
    bool failed = false;

    for (unsigned node : order) {
      // A node may be several operations -- every access to one register file
      // is bound as a unit -- so diagnostics are raised against the first of
      // them and the decision is written onto all of them.
      llvm::ArrayRef<mlir::Operation *> ops = graph.ops(node);
      mlir::Operation *rop = ops.front();
      mlir::StringAttr kind = graph.kind(node);
      if (!kind) {
        rop->emitError("design-space-exploration: operation carries no `kind`, "
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
        rop->emitError() << "design-space-exploration: the architecture "
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
        mlir::InFlightDiagnostic diag = rop->emitError()
            << "design-space-exploration: the architecture allocates "
            << pool.size() << " resource(s) of kind '" << kind.getValue()
            << "', but this conflicts with all of them -- the program needs "
               "more than the design has";
        if (mlir::FlatSymbolRefAttr storage = graph.storage(node)) {
          diag << " (storage @" << storage.getValue() << ")";
        }
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
};

} // namespace

std::unique_ptr<Binder> create_greedy_coloring_binder() {
  return std::make_unique<GreedyColoringBinder>();
}

} // namespace dse
} // namespace transformation
} // namespace vesyla
