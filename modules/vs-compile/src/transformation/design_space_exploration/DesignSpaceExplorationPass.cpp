#include "DesignSpaceExplorationPass.hpp"

#include "Architecture.hpp"
#include "ConflictGraph.hpp"
#include "Strategy.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmAttrs.hpp"
#include "vesyla/Support/Config.hpp"
#include "vesyla/Support/GlobalVar.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

namespace vesyla::transformation::dse {
#define GEN_PASS_DEF_DESIGNSPACEEXPLORATIONPASS
#include "transformation/design_space_exploration/Passes.hpp.inc"

namespace {

// Does this operation carry a resource downstream can use? Either form is
// accepted: a single endpoint, which is what an operation with one port has,
// or the [results..., operands...] array an operation with several endpoints
// needs. GenerateIcdepPass reads both.
bool has_resource(mlir::Operation *op) {
  mlir::Attribute attr = op->getAttr("resource");
  if (mlir::isa_and_nonnull<pasm::ResourceAttr>(attr)) {
    return true;
  }
  auto array = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr);
  if (!array || array.empty()) {
    return false;
  }
  return llvm::all_of(array, [](mlir::Attribute entry) {
    return mlir::isa<pasm::ResourceAttr>(entry);
  });
}

// Write the conflict graph as Graphviz DOT under the "conflict_graph_dir"
// output path. This is the pass's own reasoning made readable: what it thought
// could not be put together, and why. Losing the dump is not worth failing a
// compile over, so problems here are warnings.
void dump_conflict_graph(const ConflictGraph &graph) {
  std::string output_dir;
  if (!::vesyla::util::GlobalVar::gets("__OUTPUT_DIR__", output_dir) ||
      output_dir.empty()) {
    output_dir = ".";
  }
  ::vesyla::pasm::Config cfg;
  std::string graph_dir =
      output_dir + "/" + cfg.output_path("conflict_graph_dir");

  std::error_code dir_ec;
  std::filesystem::create_directories(graph_dir, dir_ec);
  if (dir_ec) {
    llvm::errs() << "Warning: could not create conflict graph directory "
                 << graph_dir << ": " << dir_ec.message() << "\n";
    return;
  }

  std::string path = graph_dir + "/conflict.dot";
  std::error_code file_ec;
  llvm::raw_fd_ostream os(path, file_ec, llvm::sys::fs::OF_Text);
  if (file_ec) {
    llvm::errs() << "Warning: could not open conflict graph file " << path
                 << ": " << file_ec.message() << "\n";
    return;
  }
  graph.write_dot(os, "program");
}

// The check the pass ends on: nothing may leave here unbound.
//
// Every drra.rop is walked rather than stopping at the first gap, so one run
// reports every operation that needs attention instead of one per rebuild.
mlir::LogicalResult verify_all_bound(mlir::ModuleOp module) {
  bool ok = true;
  module.walk([&](drra::RopOp rop) {
    if (has_resource(rop)) {
      return;
    }
    mlir::InFlightDiagnostic diag =
        rop.emitError("design-space-exploration: operation was not bound to a "
                      "resource");
    if (auto id = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("id")) {
      diag << " (id @" << id.getValue() << ")";
    }
    if (auto kind = rop->getAttrOfType<mlir::StringAttr>("kind")) {
      diag << " (kind '" << kind.getValue() << "')";
    }
    ok = false;
  });
  return mlir::success(ok);
}

// The other check the pass ends on: one storage, one instance.
//
// Every access to a register file has to land on the register file that access
// is in, which is what the conflict graph merges storages for. It is worth
// checking rather than assuming, because a storage that gets two instances
// reads as a working binding and fails much later as a read of a register file
// nothing ever wrote -- a program that produces zeroes rather than a compiler
// that says anything.
mlir::LogicalResult verify_storage_bindings(mlir::ModuleOp module) {
  // Where each storage was bound, and by which operation, so a disagreement can
  // name the two that disagree.
  llvm::DenseMap<mlir::Attribute, std::pair<pasm::ResourceAttr, mlir::Operation *>>
      bound;
  bool ok = true;

  module.walk([&](drra::RopOp rop) {
    auto storage = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("storage");
    if (!storage) {
      return;
    }
    // Any endpoint will do: they are all on one instance, spread over the slots
    // it occupies, and a register file access has one anyway.
    auto resource = rop->getAttrOfType<pasm::ResourceAttr>("resource");
    if (!resource) {
      if (auto array = rop->getAttrOfType<mlir::ArrayAttr>("resource");
          array && !array.empty()) {
        resource = mlir::dyn_cast<pasm::ResourceAttr>(array[0]);
      }
    }
    if (!resource) {
      return;
    }

    auto [entry, fresh] = bound.try_emplace(storage, resource, rop.getOperation());
    if (fresh) {
      return;
    }
    pasm::ResourceAttr first = entry->second.first;
    if (first.getRow() == resource.getRow() &&
        first.getCol() == resource.getCol() &&
        first.getSlot() == resource.getSlot()) {
      return;
    }
    mlir::InFlightDiagnostic diag = rop.emitError(
        "design-space-exploration: storage @");
    diag << storage.getValue() << " was bound to (" << resource.getRow() << ", "
         << resource.getCol() << ", " << resource.getSlot()
         << ") here, and to (" << first.getRow() << ", " << first.getCol()
         << ", " << first.getSlot()
         << ") elsewhere -- every access to one register file has to land on "
            "one register file";
    diag.attachNote(entry->second.second->getLoc())
        << "design-space-exploration: the other binding of @"
        << storage.getValue();
    ok = false;
  });
  return mlir::success(ok);
}

class DesignSpaceExplorationPass
    : public impl::DesignSpaceExplorationPassBase<DesignSpaceExplorationPass> {
public:
  using impl::DesignSpaceExplorationPassBase<
      DesignSpaceExplorationPass>::DesignSpaceExplorationPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    // The conflict graph comes first, and is written out before anything here
    // can fail. It says which operations cannot be put on the same instance,
    // which is what makes binding a question with an answer -- and it is read
    // off the program alone, owing nothing to the architecture. So it is still
    // worth having when the architecture turns out to be unreadable or too
    // small for the program, which is exactly when someone wants to look at it.
    //
    // One graph for the program, not one per epoch: a register file keeps what
    // was put into it, so which instance a storage lands on is a question the
    // whole program asks at once. See ConflictGraph::build.
    ConflictGraph graph = ConflictGraph::build(module);
    dump_conflict_graph(graph);

    // Allocation. Read, not decided: the architecture file passed to the
    // executable already fixes which resources exist and where.
    mlir::FailureOr<Architecture> arch = Architecture::from_config(module);
    if (mlir::failed(arch)) {
      return signalPassFailure();
    }

    // Binding: the colouring of the graph against the instances allocated.
    std::unique_ptr<Binder> binder = create_greedy_coloring_binder();
    if (mlir::failed(binder->bind(graph, *arch))) {
      return signalPassFailure();
    }

    // Both checks run, and both report everything they find: one rebuild should
    // show every operation that needs attention rather than the first.
    const bool all_bound = mlir::succeeded(verify_all_bound(module));
    const bool storage_consistent =
        mlir::succeeded(verify_storage_bindings(module));
    if (!all_bound || !storage_consistent) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::transformation::dse
