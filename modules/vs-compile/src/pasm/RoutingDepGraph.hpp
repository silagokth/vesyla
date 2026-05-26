#ifndef __VESYLA_PASM_ROUTING_DEP_GRAPH_HPP__
#define __VESYLA_PASM_ROUTING_DEP_GRAPH_HPP__

#include "Attrs.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vesyla::pasm {

enum class NodeKind { First, Last };

struct Anchor {
  mlir::FlatSymbolRefAttr instr_id;
  std::string event;
  std::vector<uint32_t> indices;
  int32_t delay;

  bool operator<(const Anchor &o) const;
  bool operator==(const Anchor &o) const;
};

using NodeKey = std::pair<int, NodeKind>;

struct Node {
  int id;
  Anchor anchor;
  NodeKind kind;
  std::string dir;
  ResourceAttr src;
  mlir::ArrayAttr dst;
  NodeKey key() const { return {id, kind}; }
};

struct Edge {
  NodeKey from;
  NodeKey to;
  bool bidir = false;
};

class RoutingDepGraph {
public:
  // Seeds the graph with start and end sentinel nodes (id=0) and an edge
  // between them.
  RoutingDepGraph();

  void insert_node(const Anchor &anchor, int id, NodeKind kind,
                   llvm::StringRef dir = {}, ResourceAttr src = {},
                   mlir::ArrayAttr dst = {});
  void insert_edge(NodeKey from, NodeKey to);
  void remove_edge(NodeKey from, NodeKey to);

  const Node *find_by_anchor(const Anchor &anchor) const;

  bool has_incoming(const Node &n) const;
  bool has_outgoing(const Node &n) const;
  bool is_bidir(NodeKey a, NodeKey b) const;

  std::vector<Node> children_of(NodeKey k) const;

  std::size_t edge_count() const { return edges_.size(); }

  void transitive_reduce();

  void dump_dot(const std::string &path) const;

  std::vector<Node>::const_iterator begin() const { return nodes_.begin(); }
  std::vector<Node>::const_iterator end() const { return nodes_.end(); }

private:
  void rebuild_adjacency();
  NodeKey edge_child(std::size_t edge_idx, NodeKey parent) const;

  std::vector<Node> nodes_;
  std::map<Anchor, std::size_t> by_anchor_;
  std::map<NodeKey, std::size_t> by_key_;
  std::vector<Edge> edges_;
  std::map<NodeKey, std::vector<std::size_t>> outgoing_;
  std::map<NodeKey, std::vector<std::size_t>> incoming_;
};

} // namespace vesyla::pasm

#endif // __VESYLA_PASM_ROUTING_DEP_GRAPH_HPP__
