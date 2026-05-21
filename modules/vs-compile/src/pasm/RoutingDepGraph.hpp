#ifndef __VESYLA_PASM_ROUTING_DEP_GRAPH_HPP__
#define __VESYLA_PASM_ROUTING_DEP_GRAPH_HPP__

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
};

using NodeKey = std::pair<int, NodeKind>;

struct Node {
  int id;
  Anchor anchor;
  NodeKind kind;
  NodeKey key() const { return {id, kind}; }
};

struct Edge {
  NodeKey from;
  NodeKey to;
};

class RoutingDepGraph {
public:
  void insert_node(const Anchor &anchor, int id, NodeKind kind);
  void insert_edge(NodeKey from, NodeKey to);

  const Node *find_by_anchor(const Anchor &anchor) const;

  bool has_incoming(const Node &n) const;
  bool has_outgoing(const Node &n) const;

  void transitive_reduce();

  void dump_dot(const std::string &path) const;

  std::vector<Node>::const_iterator begin() const { return nodes_.begin(); }
  std::vector<Node>::const_iterator end() const { return nodes_.end(); }

private:
  std::vector<Node> nodes_;
  std::map<Anchor, std::size_t> by_anchor_;
  std::map<NodeKey, std::size_t> by_key_;
  std::vector<Edge> edges_;
  std::map<NodeKey, std::vector<std::size_t>> outgoing_;
  std::map<NodeKey, std::vector<std::size_t>> incoming_;
};

} // namespace vesyla::pasm

#endif // __VESYLA_PASM_ROUTING_DEP_GRAPH_HPP__
