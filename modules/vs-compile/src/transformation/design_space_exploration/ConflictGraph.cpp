#include "ConflictGraph.hpp"

#include "mlir/Analysis/Liveness.h"
#include "llvm/ADT/STLExtras.h"

#include "vesyla/Dialect/Drra/IR/DrraOps.hpp"
#include "vesyla/Dialect/Pasm/IR/PasmOps.hpp"

#include <string>

namespace vesyla {
namespace transformation {
namespace dse {

namespace {

using OccupancyMap =
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned>>;

// Node `index` claims an instance at `op` and, when `op` has regions,
// everywhere inside it.
//
// A value live across an operation with regions is live across everything that
// operation runs, so an affine.for a value is carried over is claimed body and
// all. Without this an operation inside the loop would look free to take the
// instance the carried value is sitting in.
void occupy(OccupancyMap &occupancy, mlir::Operation *op, unsigned index) {
  op->walk([&](mlir::Operation *nested) { occupancy[nested].push_back(index); });
}

// The parts of its resource the operation says it holds, in declaration order.
//
// Anything that is not an array of strings is read as an empty list rather than
// reported: the array is written by the component library, and the selector
// checks its shape when it loads the pattern, so by the time it reaches here it
// has already been vetted. Empty means the operation declared nothing, which
// callers read as holding the whole resource.
llvm::SmallVector<mlir::StringAttr> read_parts(mlir::Operation *op) {
  llvm::SmallVector<mlir::StringAttr> parts;
  auto array = op->getAttrOfType<mlir::ArrayAttr>("uses");
  if (!array) {
    return parts;
  }
  for (mlir::Attribute entry : array) {
    if (auto part = mlir::dyn_cast<mlir::StringAttr>(entry)) {
      parts.push_back(part);
    }
  }
  return parts;
}

// The parts both lists name. Kept in `a`'s declaration order rather than
// sorted, because the alternative -- ordering by the attributes' addresses --
// would put the same graph's edge labels in a different order on every run.
// Both lists are a handful of entries, so the quadratic scan does not matter.
llvm::SmallVector<mlir::StringAttr>
intersect(llvm::ArrayRef<mlir::StringAttr> a,
          llvm::ArrayRef<mlir::StringAttr> b) {
  llvm::SmallVector<mlir::StringAttr> shared;
  for (mlir::StringAttr part : a) {
    if (llvm::is_contained(b, part) && !llvm::is_contained(shared, part)) {
      shared.push_back(part);
    }
  }
  return shared;
}

} // namespace

ConflictGraph ConflictGraph::build(mlir::ModuleOp module) {
  ConflictGraph graph;
  mlir::Operation *scope = module.getOperation();

  // The nodes. Operations that hold the same storage on the same kind are one
  // node, because they have to be bound together; everything else is a node of
  // its own. Grouping on the kind as well as the storage keeps a node from
  // being asked to be two kinds of instance at once, which could otherwise
  // happen where two resources match accesses to one buffer.
  // Keyed on the generic Attribute: DenseMapInfo for a concrete attribute class
  // hands back its base, which will not convert to the derived type.
  llvm::DenseMap<std::pair<mlir::Attribute, mlir::Attribute>, unsigned>
      by_storage;
  scope->walk([&](drra::RopOp rop) {
    mlir::Operation *op = rop.getOperation();
    auto kind = rop->getAttrOfType<mlir::StringAttr>("kind");
    auto storage = rop->getAttrOfType<mlir::FlatSymbolRefAttr>("storage");

    unsigned index = graph.nodes_.size();
    if (kind && storage) {
      auto [entry, fresh] = by_storage.try_emplace({kind, storage}, index);
      index = entry->second;
      if (fresh) {
        graph.nodes_.push_back(Node{{}, kind, storage, {}});
      }
    } else {
      graph.nodes_.push_back(Node{{}, kind, storage, {}});
    }

    Node &node = graph.nodes_[index];
    node.ops.push_back(op);
    for (mlir::StringAttr part : read_parts(op)) {
      if (!llvm::is_contained(node.parts, part)) {
        node.parts.push_back(part);
      }
    }
    graph.node_index_[op] = index;
  });
  graph.adjacency_.resize(graph.nodes_.size());
  if (graph.nodes_.empty()) {
    return graph;
  }

  // Different storage on one kind. One instance holds one storage, so these
  // conflict whether or not they ever run at the same time -- which is why the
  // pairs are walked here rather than left to the occupancies below.
  for (unsigned a = 0; a < graph.nodes_.size(); ++a) {
    if (!graph.nodes_[a].storage) {
      continue;
    }
    for (unsigned b = a + 1; b < graph.nodes_.size(); ++b) {
      if (!graph.nodes_[b].storage ||
          graph.nodes_[a].storage == graph.nodes_[b].storage) {
        continue;
      }
      graph.add_edge(a, b, ConflictReason::distinct_storage);
    }
  }

  // Data flow. A producer still holds its value when its consumer reads it, so
  // the two cannot be the same instance. Liveness below finds most of these on
  // its own; drawing them outright keeps the edge that matters most from
  // resting on how the analysis treats the boundary between the two.
  for (unsigned i = 0; i < graph.nodes_.size(); ++i) {
    for (mlir::Operation *op : graph.nodes_[i].ops) {
      for (mlir::Value operand : op->getOperands()) {
        mlir::Operation *producer = operand.getDefiningOp();
        if (!producer) {
          continue;
        }
        auto found = graph.node_index_.find(producer);
        if (found != graph.node_index_.end()) {
          graph.add_edge(i, found->second, ConflictReason::data_flow);
        }
      }
    }
  }

  // Occupancy. Where each node lays claim to an instance: each of its
  // operations, plus everywhere one of their results is live.
  mlir::Liveness liveness(scope);
  OccupancyMap occupancy;
  for (unsigned i = 0; i < graph.nodes_.size(); ++i) {
    for (mlir::Operation *op : graph.nodes_[i].ops) {
      occupancy[op].push_back(i);
      for (mlir::Value result : op->getResults()) {
        for (mlir::Operation *live_op : liveness.resolveLiveness(result)) {
          occupy(occupancy, live_op, i);
        }
      }
    }
  }

  // Two nodes claiming an instance at the same point may still not contend --
  // add_edge has the last word on that, since it is what knows which parts of
  // the resource each of them wanted. Walking the claims per point rather than
  // comparing every pair of nodes keeps this to the size of the overlaps that
  // actually exist.
  for (auto &entry : occupancy) {
    llvm::SmallVector<unsigned> &here = entry.second;
    llvm::sort(here);
    here.erase(llvm::unique(here), here.end());
    for (unsigned a = 0; a < here.size(); ++a) {
      for (unsigned b = a + 1; b < here.size(); ++b) {
        graph.add_edge(here[a], here[b], ConflictReason::live_overlap);
      }
    }
  }

  return graph;
}

void ConflictGraph::add_edge(unsigned a, unsigned b, ConflictReason why) {
  if (a == b) {
    return;
  }
  // Different kinds never land on one instance, so there is nothing to say
  // about them; a node whose kind selection did not write has nothing to
  // compare against either.
  if (!nodes_[a].kind || nodes_[a].kind != nodes_[b].kind) {
    return;
  }

  EdgeKey key = edge_key(a, b);
  auto found = edges_.find(key);
  if (found == edges_.end()) {
    llvm::SmallVector<mlir::StringAttr> shared;
    bool distinct_storage = nodes_[a].storage && nodes_[b].storage &&
                            nodes_[a].storage != nodes_[b].storage;
    if (!distinct_storage && !nodes_[a].parts.empty() &&
        !nodes_[b].parts.empty()) {
      // Same resource, same moment -- but a resource has parts, and two nodes
      // only contend when they want the same one. A node that declared no parts
      // is read as holding all of them, so it contends with everything of its
      // kind.
      shared = intersect(nodes_[a].parts, nodes_[b].parts);
      if (shared.empty()) {
        return;
      }
    }
    found = edges_.insert({key, Edge{ConflictReason::none, std::move(shared)}})
                .first;
    adjacency_[a].insert(b);
    adjacency_[b].insert(a);
  }
  found->second.why = found->second.why | why;
}

ConflictReason ConflictGraph::reason(unsigned a, unsigned b) const {
  auto found = edges_.find(edge_key(a, b));
  return found == edges_.end() ? ConflictReason::none : found->second.why;
}

llvm::ArrayRef<mlir::StringAttr>
ConflictGraph::shared_parts(unsigned a, unsigned b) const {
  auto found = edges_.find(edge_key(a, b));
  if (found == edges_.end()) {
    return {};
  }
  return found->second.shared;
}

std::optional<unsigned> ConflictGraph::node_of(mlir::Operation *op) const {
  auto found = node_index_.find(op);
  if (found == node_index_.end()) {
    return std::nullopt;
  }
  return found->second;
}

namespace {

// The operation's `id` symbol, or its position when selection gave it none,
// followed by the epoch it sits in.
//
// The epoch is worth the room: a node holding storage gathers every access to
// it, and those run in whichever epochs touch that register file. Which epoch
// each access is in is the thing the label would otherwise lose, and it is
// what makes a storage node read as the load in one epoch and the use in the
// next rather than as an undifferentiated list.
std::string op_name(mlir::Operation *op, unsigned fallback) {
  std::string name;
  if (auto id = op->getAttrOfType<mlir::FlatSymbolRefAttr>("id")) {
    name = id.getValue().str();
  } else {
    name = "#" + std::to_string(fallback);
  }
  if (auto epoch = op->getParentOfType<pasm::EpochOp>()) {
    name += "(" + epoch.getId().str() + ")";
  }
  return name;
}

std::string join(llvm::ArrayRef<mlir::StringAttr> parts) {
  std::string text;
  for (unsigned i = 0; i < parts.size(); ++i) {
    text += (i == 0 ? "" : ", ") + parts[i].getValue().str();
  }
  return text;
}

} // namespace

std::string ConflictGraph::node_label(unsigned node) const {
  const Node &n = nodes_[node];
  std::string label = std::to_string(node) + ": ";
  // A node that holds storage is named for it, because that is what is being
  // bound; the accesses that make it up go underneath.
  if (n.storage) {
    label += n.storage.getValue().str();
  } else {
    label += op_name(n.ops.front(), node);
  }
  if (n.ops.size() > 1 || n.storage) {
    label += "\\n";
    for (unsigned i = 0; i < n.ops.size(); ++i) {
      label += (i == 0 ? "" : ", ") + op_name(n.ops[i], node);
    }
  }
  // The parts go on their own line, so a node says what it holds and not only
  // what it is. A node that declared none says so, since that is why it ends up
  // conflicting with everything of its kind.
  label += "\\n";
  label += n.parts.empty() ? "(whole resource)" : join(n.parts);
  return label;
}

void ConflictGraph::write_dot(llvm::raw_ostream &os,
                              llvm::StringRef graph_name) const {
  os << "graph \"" << graph_name << "\" {\n";
  os << "  node [shape=box];\n";

  // Nodes grouped by kind, which is also how they are grouped when bound: a
  // cluster is the set of nodes competing for the same pool of instances.
  llvm::SmallVector<mlir::StringAttr> kinds_seen;
  for (const Node &node : nodes_) {
    if (node.kind && !llvm::is_contained(kinds_seen, node.kind)) {
      kinds_seen.push_back(node.kind);
    }
  }
  for (mlir::StringAttr kind : kinds_seen) {
    os << "  subgraph \"cluster_" << kind.getValue() << "\" {\n";
    os << "    label=\"" << kind.getValue() << "\";\n";
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].kind == kind) {
        os << "    n" << i << " [label=\"" << node_label(i) << "\"];\n";
      }
    }
    os << "  }\n";
  }
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    if (!nodes_[i].kind) {
      os << "  n" << i << " [label=\"" << node_label(i)
         << "\", style=dashed, color=gray];\n";
    }
  }

  // An edge is labelled with what the two contend for, which is the part of
  // the answer worth reading, and under it the reason they were compared at
  // all.
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    for (unsigned j : adjacency_[i]) {
      if (i >= j) {
        continue;
      }
      ConflictReason why = reason(i, j);
      llvm::ArrayRef<mlir::StringAttr> shared = shared_parts(i, j);

      std::string label;
      if (!shared.empty()) {
        label = join(shared);
      } else if (has_reason(why, ConflictReason::distinct_storage)) {
        label = "different storage";
      } else {
        label = "whole resource";
      }

      std::string reasons;
      for (auto [bit, text] :
           {std::pair{ConflictReason::distinct_storage, "storage"},
            std::pair{ConflictReason::data_flow, "data flow"},
            std::pair{ConflictReason::live_overlap, "live"}}) {
        if (has_reason(why, bit)) {
          reasons += reasons.empty() ? text : std::string(" + ") + text;
        }
      }
      os << "  n" << i << " -- n" << j << " [label=\"" << label << "\\n("
         << reasons << ")\"];\n";
    }
  }
  os << "}\n";
}

} // namespace dse
} // namespace transformation
} // namespace vesyla
