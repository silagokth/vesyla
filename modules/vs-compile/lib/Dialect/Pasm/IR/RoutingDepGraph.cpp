#include "vesyla/Dialect/Pasm/IR/RoutingDepGraph.hpp"

#include "vesyla/Support/Anchor.hpp"

#include <fstream>
#include <queue>
#include <set>
#include <utility>

namespace vesyla::pasm {

namespace {

const char *kind_str(NodeKind k) {
  return k == NodeKind::First ? "first" : "last";
}

std::string node_dot_id(int id, NodeKind kind) {
  return "n" + std::to_string(id) + "_" + kind_str(kind);
}

// Escape characters that are structural in a DOT record label so user-supplied
// text doesn't accidentally break the layout. Newlines become the DOT "\n"
// escape so they render as line breaks inside the label.
std::string escape_dot_label(llvm::StringRef in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    switch (c) {
    case '"':
    case '\\':
    case '|':
    case '<':
    case '>':
    case '{':
    case '}':
      out.push_back('\\');
      out.push_back(c);
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out.push_back(c);
    }
  }
  return out;
}

} // namespace

bool Anchor::operator<(const Anchor &o) const {
  llvm::StringRef a = instr_id ? instr_id.getValue() : llvm::StringRef();
  llvm::StringRef b = o.instr_id ? o.instr_id.getValue() : llvm::StringRef();
  if (a != b) {
    return a.compare(b) < 0;
  }
  if (or_idx != o.or_idx) {
    return or_idx < o.or_idx;
  }
  if (mt != o.mt) {
    return mt < o.mt;
  }
  if (ir_idx != o.ir_idx) {
    return ir_idx < o.ir_idx;
  }
  return delay < o.delay;
}

bool Anchor::operator==(const Anchor &o) const {
  llvm::StringRef a = instr_id ? instr_id.getValue() : llvm::StringRef();
  llvm::StringRef b = o.instr_id ? o.instr_id.getValue() : llvm::StringRef();
  return a == b && or_idx == o.or_idx && mt == o.mt && ir_idx == o.ir_idx &&
         delay == o.delay;
}

RoutingDepGraph::RoutingDepGraph() {
  Anchor sentinel{};
  insert_node(sentinel, 0, NodeKind::First);
  insert_node(sentinel, 0, NodeKind::Last);
  insert_edge({0, NodeKind::First}, {0, NodeKind::Last});
}

void RoutingDepGraph::insert_node(const Anchor &anchor, int id, NodeKind kind,
                                  llvm::StringRef dir, ResourceAttr src,
                                  mlir::ArrayAttr dst) {
  std::size_t idx = nodes_.size();
  nodes_.push_back(Node{id, anchor, kind, dir.str(), src, dst});
  by_anchor_[anchor] = idx;
  by_key_[{id, kind}] = idx;
}

void RoutingDepGraph::rebuild_adjacency() {
  outgoing_.clear();
  incoming_.clear();
  for (std::size_t i = 0; i < edges_.size(); ++i) {
    outgoing_[edges_[i].from].push_back(i);
    incoming_[edges_[i].to].push_back(i);
    if (edges_[i].bidir) {
      outgoing_[edges_[i].to].push_back(i);
      incoming_[edges_[i].from].push_back(i);
    }
  }
}

NodeKey RoutingDepGraph::edge_child(std::size_t edge_idx, NodeKey parent) const {
  const Edge &e = edges_[edge_idx];
  if (e.bidir && e.to == parent) {
    return e.from;
  }
  return e.to;
}

std::vector<Node> RoutingDepGraph::children_of(NodeKey k) const {
  std::vector<Node> out;
  auto it = outgoing_.find(k);
  if (it == outgoing_.end()) {
    return out;
  }
  for (std::size_t idx : it->second) {
    NodeKey child = edge_child(idx, k);
    auto kit = by_key_.find(child);
    if (kit != by_key_.end()) {
      out.push_back(nodes_[kit->second]);
    }
  }
  return out;
}

void RoutingDepGraph::remove_edge(NodeKey from, NodeKey to) {
  std::size_t target = edges_.size();
  for (std::size_t i = 0; i < edges_.size(); ++i) {
    if (edges_[i].from == from && edges_[i].to == to) {
      target = i;
      break;
    }
    if (edges_[i].bidir && edges_[i].from == to && edges_[i].to == from) {
      target = i;
      break;
    }
  }
  if (target == edges_.size()) {
    return;
  }
  edges_.erase(edges_.begin() + target);
  rebuild_adjacency();
}

void RoutingDepGraph::insert_edge(NodeKey from, NodeKey to) {
  // Skip if from->to already exists (direct or via bidir).
  auto it = outgoing_.find(from);
  if (it != outgoing_.end()) {
    for (std::size_t i : it->second) {
      if (edge_child(i, from) == to) {
        return;
      }
    }
  }

  // If reverse edge to->from exists, upgrade it to bidir.
  auto rev_it = outgoing_.find(to);
  if (rev_it != outgoing_.end()) {
    for (std::size_t i : rev_it->second) {
      if (!edges_[i].bidir && edges_[i].from == to && edges_[i].to == from) {
        edges_[i].bidir = true;
        outgoing_[from].push_back(i);
        incoming_[to].push_back(i);
        return;
      }
    }
  }

  std::size_t idx = edges_.size();
  edges_.push_back(Edge{from, to, false});
  outgoing_[from].push_back(idx);
  incoming_[to].push_back(idx);
}

void RoutingDepGraph::transitive_reduce() {
  std::vector<bool> removed(edges_.size(), false);

  auto reachable = [&](NodeKey src, NodeKey target, std::size_t skip_edge) {
    std::set<NodeKey> seen;
    std::queue<NodeKey> q;
    seen.insert(src);
    q.push(src);
    while (!q.empty()) {
      NodeKey u = q.front();
      q.pop();
      auto it = outgoing_.find(u);
      if (it == outgoing_.end()) {
        continue;
      }
      for (std::size_t i : it->second) {
        if (i == skip_edge || removed[i]) {
          continue;
        }
        NodeKey v = edge_child(i, u);
        if (v == target) {
          return true;
        }
        if (seen.insert(v).second) {
          q.push(v);
        }
      }
    }
    return false;
  };

  NodeKey start{0, NodeKind::First};
  std::set<NodeKey> visited;
  std::queue<NodeKey> q;
  visited.insert(start);
  q.push(start);
  while (!q.empty()) {
    NodeKey u = q.front();
    q.pop();
    auto it = outgoing_.find(u);
    if (it == outgoing_.end()) {
      continue;
    }
    std::vector<std::size_t> outs(it->second.begin(), it->second.end());
    for (std::size_t edge_idx : outs) {
      NodeKey v = edge_child(edge_idx, u);
      if (!removed[edge_idx] && !edges_[edge_idx].bidir &&
          reachable(u, v, edge_idx)) {
        removed[edge_idx] = true;
      }
      if (visited.insert(v).second) {
        q.push(v);
      }
    }
  }

  std::vector<Edge> new_edges;
  new_edges.reserve(edges_.size());
  for (std::size_t i = 0; i < edges_.size(); ++i) {
    if (!removed[i]) {
      new_edges.push_back(edges_[i]);
    }
  }
  edges_ = std::move(new_edges);
  rebuild_adjacency();
}

const Node *RoutingDepGraph::find_by_anchor(const Anchor &anchor) const {
  auto it = by_anchor_.find(anchor);
  if (it == by_anchor_.end()) {
    return nullptr;
  }
  return &nodes_[it->second];
}

bool RoutingDepGraph::has_incoming(const Node &n) const {
  auto it = incoming_.find(n.key());
  if (it == incoming_.end()) {
    return false;
  }
  for (std::size_t idx : it->second) {
    if (!edges_[idx].bidir) {
      return true;
    }
  }
  return false;
}

bool RoutingDepGraph::has_outgoing(const Node &n) const {
  auto it = outgoing_.find(n.key());
  if (it == outgoing_.end()) {
    return false;
  }
  for (std::size_t idx : it->second) {
    if (!edges_[idx].bidir) {
      return true;
    }
  }
  return false;
}

bool RoutingDepGraph::is_bidir(NodeKey a, NodeKey b) const {
  auto it = outgoing_.find(a);
  if (it == outgoing_.end()) {
    return false;
  }
  for (std::size_t i : it->second) {
    if (edges_[i].bidir && edge_child(i, a) == b) {
      return true;
    }
  }
  return false;
}

void RoutingDepGraph::dump_dot(const std::string &path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    return;
  }

  ofs << "digraph RoutingDepGraph {\n";
  ofs << "  rankdir=TB;\n";
  ofs << "  { rank=source; " << node_dot_id(0, NodeKind::First) << "; }\n";
  ofs << "  { rank=sink; " << node_dot_id(0, NodeKind::Last) << "; }\n";

  for (const Node &n : nodes_) {
    std::string id = node_dot_id(n.id, n.kind);
    if (n.id == 0 && n.kind == NodeKind::First) {
      ofs << "  " << id << " [shape=circle, label=\"start\"];\n";
    } else if (n.id == 0 && n.kind == NodeKind::Last) {
      ofs << "  " << id << " [shape=circle, label=\"end\"];\n";
    } else {
      llvm::StringRef instr_name =
          n.anchor.instr_id ? n.anchor.instr_id.getValue() : llvm::StringRef();
      std::string header = std::to_string(n.id) + " " + kind_str(n.kind);
      if (!n.dir.empty()) {
        header += " (" + n.dir + ")";
      }
      std::string route;
      if (n.src) {
        route = std::to_string(n.src.getSlot());
        if (n.dst) {
          for (mlir::Attribute attr : n.dst) {
            auto dst_res = mlir::dyn_cast<ResourceAttr>(attr);
            if (dst_res) {
              route += " -\\> " + std::to_string(dst_res.getSlot());
            }
          }
        }
      }
      ::vesyla::Anchor va;
      va.or_idx.assign(n.anchor.or_idx.begin(), n.anchor.or_idx.end());
      va.mt_idx = static_cast<int>(n.anchor.mt);
      va.ir_idx.assign(n.anchor.ir_idx.begin(), n.anchor.ir_idx.end());
      std::string body = "instr: " + escape_dot_label(instr_name) +
                         "\\nanchor: " + escape_dot_label(va.to_string()) +
                         "\\ndelay: " + std::to_string(n.anchor.delay);
      if (!route.empty()) {
        body += "\\nslot: " + route;
      }
      ofs << "  " << id << " [shape=record, label=\"{"
          << escape_dot_label(header) << "|" << body << "}\"];\n";
    }
  }

  for (const Edge &e : edges_) {
    std::string a = node_dot_id(e.from.first, e.from.second);
    std::string b = node_dot_id(e.to.first, e.to.second);
    if (e.bidir) {
      ofs << "  " << a << " -> " << b << " [dir=both];\n";
    } else {
      ofs << "  " << a << " -> " << b << ";\n";
    }
  }

  ofs << "}\n";
}

} // namespace vesyla::pasm
