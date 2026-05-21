#include "RoutingDepGraph.hpp"

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

std::string indices_to_str(const std::vector<uint32_t> &indices) {
  std::string s = "[";
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += std::to_string(indices[i]);
  }
  s += "]";
  return s;
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
  if (event != o.event) {
    return event < o.event;
  }
  if (indices != o.indices) {
    return indices < o.indices;
  }
  return delay < o.delay;
}

void RoutingDepGraph::insert_node(const Anchor &anchor, int id, NodeKind kind) {
  std::size_t idx = nodes_.size();
  nodes_.push_back(Node{id, anchor, kind});
  by_anchor_[anchor] = idx;
  by_key_[{id, kind}] = idx;
}

void RoutingDepGraph::insert_edge(NodeKey from, NodeKey to) {
  // Skip if a direct edge from->to already exists.
  auto it = outgoing_.find(from);
  if (it != outgoing_.end()) {
    for (std::size_t i : it->second) {
      if (edges_[i].to == to) {
        return;
      }
    }
  }

  std::size_t idx = edges_.size();
  edges_.push_back(Edge{from, to});
  outgoing_[from].push_back(idx);
  incoming_[to].push_back(idx);
}

void RoutingDepGraph::transitive_reduce() {
  std::vector<bool> remove_flag(edges_.size(), false);

  for (const Node &u : nodes_) {
    NodeKey u_key = u.key();
    auto out_it = outgoing_.find(u_key);
    if (out_it == outgoing_.end()) {
      continue;
    }

    std::vector<std::size_t> direct_edges;
    std::vector<NodeKey> frontier;
    for (std::size_t i : out_it->second) {
      direct_edges.push_back(i);
      frontier.push_back(edges_[i].to);
    }

    // BFS from frontier, treating u as blocked. Each node is visited at most
    // once. Anything reached this way is reachable from u via a path of
    // length >= 2.
    std::set<NodeKey> reachable;
    std::set<NodeKey> visited;
    visited.insert(u_key);
    std::queue<NodeKey> q;
    for (NodeKey f : frontier) {
      if (visited.insert(f).second) {
        q.push(f);
      }
    }
    while (!q.empty()) {
      NodeKey x = q.front();
      q.pop();
      auto ox = outgoing_.find(x);
      if (ox == outgoing_.end()) {
        continue;
      }
      for (std::size_t i : ox->second) {
        NodeKey w = edges_[i].to;
        if (w == u_key) {
          continue;
        }
        reachable.insert(w);
        if (visited.insert(w).second) {
          q.push(w);
        }
      }
    }

    for (std::size_t i : direct_edges) {
      if (reachable.count(edges_[i].to)) {
        remove_flag[i] = true;
      }
    }
  }

  std::vector<Edge> new_edges;
  new_edges.reserve(edges_.size());
  for (std::size_t i = 0; i < edges_.size(); ++i) {
    if (!remove_flag[i]) {
      new_edges.push_back(edges_[i]);
    }
  }
  edges_ = std::move(new_edges);
  outgoing_.clear();
  incoming_.clear();
  for (std::size_t i = 0; i < edges_.size(); ++i) {
    outgoing_[edges_[i].from].push_back(i);
    incoming_[edges_[i].to].push_back(i);
  }
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
  auto out_it = outgoing_.find(n.key());
  for (std::size_t idx : it->second) {
    NodeKey from = edges_[idx].from;
    bool has_reverse = false;
    if (out_it != outgoing_.end()) {
      for (std::size_t j : out_it->second) {
        if (edges_[j].to == from) {
          has_reverse = true;
          break;
        }
      }
    }
    if (!has_reverse) {
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
  auto in_it = incoming_.find(n.key());
  for (std::size_t idx : it->second) {
    NodeKey to = edges_[idx].to;
    bool has_reverse = false;
    if (in_it != incoming_.end()) {
      for (std::size_t j : in_it->second) {
        if (edges_[j].from == to) {
          has_reverse = true;
          break;
        }
      }
    }
    if (!has_reverse) {
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
      llvm::StringRef instr_name = n.anchor.instr_id
                                       ? n.anchor.instr_id.getValue()
                                       : llvm::StringRef();
      std::string header =
          std::to_string(n.id) + " " + kind_str(n.kind);
      std::string body = "instr: " + escape_dot_label(instr_name) +
                         "\\nevent: " + escape_dot_label(n.anchor.event) +
                         "\\nindices: " + indices_to_str(n.anchor.indices) +
                         "\\ndelay: " + std::to_string(n.anchor.delay);
      ofs << "  " << id << " [shape=record, label=\"{"
          << escape_dot_label(header) << "|" << body << "}\"];\n";
    }
  }

  for (const Edge &e : edges_) {
    std::string a = node_dot_id(e.from.first, e.from.second);
    std::string b = node_dot_id(e.to.first, e.to.second);
    ofs << "  " << a << " -> " << b << ";\n";
  }

  ofs << "}\n";
}

} // namespace vesyla::pasm
