#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <filesystem>
#include <fstream>

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/Transforms/RessourceBindingPass.hpp"
#include "vesyla/Support/Config.hpp"

namespace vesyla::pasm {
#define GEN_PASS_DEF_RESSOURCEBINDINGPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Undirected conflict (interference) graph over the drra.rop ops of a single
// epoch. Two nodes conflict when they are of the same resource `type` and their
// result values are simultaneously live, or when one node's result is a direct
// operand of the other. Coloring this graph is what assigns each node a concrete
// hardware resource.
struct ConflictGraph {
  llvm::SmallVector<mlir::Operation *> nodes;
  llvm::DenseMap<mlir::Operation *, unsigned> node_index;
  llvm::SmallVector<mlir::Attribute> node_type;
  llvm::SmallVector<llvm::SmallDenseSet<unsigned>> adjacency;

  void add_edge(unsigned a, unsigned b) {
    if (a == b) {
      return;
    }
    // Only nodes of the same resource type contend for a resource; a missing
    // type never conflicts.
    if (!node_type[a] || node_type[a] != node_type[b]) {
      return;
    }
    adjacency[a].insert(b);
    adjacency[b].insert(a);
  }
};

// Build the conflict graph for a single epoch.
ConflictGraph build_conflict_graph(EpochOp epoch) {
  ConflictGraph graph;

  // Step 1: collect every drra.rop in the epoch as a node, recording its
  // resource `type` (used to gate conflicts to same-type nodes).
  epoch.walk([&](drra::RopOp rop) {
    graph.node_index[rop.getOperation()] = graph.nodes.size();
    graph.nodes.push_back(rop.getOperation());
    graph.node_type.push_back(rop->getAttr("type"));
  });
  graph.adjacency.resize(graph.nodes.size());

  // Step 2: direct def-use edges. An edge exists whenever a node's result is a
  // direct operand of another node.
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    mlir::Operation *consumer = graph.nodes[i];
    for (mlir::Value operand : consumer->getOperands()) {
      mlir::Operation *producer = operand.getDefiningOp();
      if (!producer) {
        continue;
      }
      auto it = graph.node_index.find(producer);
      if (it != graph.node_index.end()) {
        graph.add_edge(i, it->second);
      }
    }
  }

  // Step 3: liveness-overlap edges. Two nodes conflict when their result
  // values are live at a common operation. resolveLiveness returns every op at
  // which a value is live, resolved across affine.for / iter_args control flow.
  mlir::Liveness liveness(epoch.getOperation());
  llvm::SmallVector<llvm::DenseSet<mlir::Operation *>> live_ops(
      graph.nodes.size());
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    for (mlir::Value result : graph.nodes[i]->getResults()) {
      for (mlir::Operation *op : liveness.resolveLiveness(result)) {
        live_ops[i].insert(op);
      }
    }
  }
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    for (unsigned j = i + 1; j < graph.nodes.size(); ++j) {
      for (mlir::Operation *op : live_ops[i]) {
        if (live_ops[j].contains(op)) {
          graph.add_edge(i, j);
          break;
        }
      }
    }
  }

  return graph;
}

// Human-readable label for a node: its index plus its drra.rop `id` symbol
// when present (e.g. "3: compute").
std::string node_label(const ConflictGraph &graph, unsigned i) {
  std::string label = std::to_string(i);
  if (auto id = graph.nodes[i]->getAttrOfType<mlir::FlatSymbolRefAttr>("id")) {
    label += ": " + id.getValue().str();
  }
  return label;
}

// Dump the conflict graph as Graphviz DOT (renderable to an image). The output
// directory is configurable via the "conflict_graph_dir" output path in the
// config file; the file name is derived from the epoch id.
void dump_conflict_graph_dot(const ConflictGraph &graph, EpochOp epoch) {
  std::string output_dir;
  if (!vesyla::util::GlobalVar::gets("__OUTPUT_DIR__", output_dir) ||
      output_dir.empty()) {
    output_dir = ".";
  }
  ::vesyla::pasm::Config cfg;
  std::string conflict_graph_dir =
      output_dir + "/" + cfg.output_path("conflict_graph_dir");
  std::error_code dir_ec;
  std::filesystem::create_directories(conflict_graph_dir, dir_ec);
  if (dir_ec) {
    llvm::errs() << "Warning: could not create conflict graph directory "
                 << conflict_graph_dir << ": " << dir_ec.message() << "\n";
    return;
  }

  std::string path =
      conflict_graph_dir + "/conflict_" + epoch.getId().str() + ".dot";
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    llvm::errs() << "Warning: could not open conflict graph file " << path
                 << "\n";
    return;
  }

  ofs << "graph conflict {\n";
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    ofs << "  n" << i << " [label=\"" << node_label(graph, i) << "\"];\n";
  }
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    for (unsigned j : graph.adjacency[i]) {
      if (i < j) {
        ofs << "  n" << i << " -- n" << j << ";\n";
      }
    }
  }
  ofs << "}\n";
  ofs.close();
}

class RessourceBindingPassRewriter : public mlir::OpRewritePattern<EpochOp> {
public:
  using mlir::OpRewritePattern<EpochOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EpochOp op, mlir::PatternRewriter &rewriter) const final {
    ConflictGraph graph = build_conflict_graph(op);
    dump_conflict_graph_dot(graph, op);
    // TODO: color the graph and assign resource attributes.

    // Nothing is mutated yet, so report failure to avoid re-triggering the
    // greedy driver on this epoch. Return success() once coloring writes the
    // resource attributes (and guard the match against already-bound epochs).
    return mlir::failure();
  }
};

class RessourceBindingPass
    : public impl::RessourceBindingPassBase<RessourceBindingPass> {
public:
  using impl::RessourceBindingPassBase<
      RessourceBindingPass>::RessourceBindingPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RessourceBindingPassRewriter>(&getContext());
    mlir::FrozenRewritePatternSet pattern_set(std::move(patterns));
    if (failed(applyPatternsGreedily(module, pattern_set))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace vesyla::pasm
