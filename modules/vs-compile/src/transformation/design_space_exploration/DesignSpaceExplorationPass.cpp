#include "DesignSpaceExplorationPass.hpp"

#include "Architecture.hpp"
#include "ConflictGraph.hpp"
#include "Strategy.hpp"

#include "mlir/IR/BuiltinAttributes.h"
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

// What the scope is called, in diagnostics and in the name of its dumped
// graph: the epoch's id, or "module" for a program with no epochs.
std::string scope_name(mlir::Operation *scope) {
  if (auto epoch = mlir::dyn_cast<pasm::EpochOp>(scope)) {
    return epoch.getId().str();
  }
  return "module";
}

// Write the conflict graph as Graphviz DOT, one file per scope, under the
// "conflict_graph_dir" output path. This is the pass's own reasoning made
// readable: what it thought could not be put together, and why. Losing the
// dump is not worth failing a compile over, so problems here are warnings.
void dump_conflict_graph(const ConflictGraph &graph, mlir::Operation *scope) {
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

  std::string name = scope_name(scope);
  std::string path = graph_dir + "/conflict_" + name + ".dot";
  std::error_code file_ec;
  llvm::raw_fd_ostream os(path, file_ec, llvm::sys::fs::OF_Text);
  if (file_ec) {
    llvm::errs() << "Warning: could not open conflict graph file " << path
                 << ": " << file_ec.message() << "\n";
    return;
  }
  graph.write_dot(os, name);
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

class DesignSpaceExplorationPass
    : public impl::DesignSpaceExplorationPassBase<DesignSpaceExplorationPass> {
public:
  using impl::DesignSpaceExplorationPassBase<
      DesignSpaceExplorationPass>::DesignSpaceExplorationPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    // Each pasm.epoch is handled on its own: an epoch is the region downstream
    // scheduling treats as a unit, so it is also the scope a solution is
    // searched over. A program with no epochs is handled as a whole.
    llvm::SmallVector<mlir::Operation *> scopes;
    module.walk([&](pasm::EpochOp epoch) { scopes.push_back(epoch); });
    if (scopes.empty()) {
      scopes.push_back(module.getOperation());
    }

    // The conflict graphs come first, and are written out before anything here
    // can fail. They say which operations cannot be put on the same instance,
    // which is what makes binding a question with an answer -- and they are
    // read off the program alone, owing nothing to the architecture. So they
    // are still worth having when the architecture turns out to be unreadable
    // or too small for the program, which is exactly when someone wants to look
    // at one.
    // An explicit inline count: ConflictGraph is past the size SmallVector will
    // pick one for on its own.
    llvm::SmallVector<ConflictGraph, 2> graphs;
    for (mlir::Operation *scope : scopes) {
      graphs.push_back(ConflictGraph::build(scope));
      dump_conflict_graph(graphs.back(), scope);
    }

    // Allocation. Read, not decided: the architecture file passed to the
    // executable already fixes which resources exist and where.
    mlir::FailureOr<Architecture> arch = Architecture::from_config(module);
    if (mlir::failed(arch)) {
      return signalPassFailure();
    }

    // Binding: the colouring of each graph against the instances allocated.
    std::unique_ptr<Binder> binder = create_greedy_coloring_binder();
    for (const ConflictGraph &graph : graphs) {
      if (mlir::failed(binder->bind(graph, *arch))) {
        return signalPassFailure();
      }
    }

    if (mlir::failed(verify_all_bound(module))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::transformation::dse
