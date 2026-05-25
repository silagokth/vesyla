#include "InterconnectBinding.hpp"

namespace vesyla::pasm {

int direction_code(int dr, int dc) {
  int sr = (dr > 0) - (dr < 0);
  int sc = (dc > 0) - (dc < 0);
  return (sr + 1) * 3 + (sc + 1);
}

void dump_binding(const InterconnectBinding &b, llvm::raw_ostream &os) {
  auto print_resource = [&](ResourceAttr r) {
    if (r) {
      os << r.getSlot();
    } else {
      os << "null";
    }
  };
  for (std::size_t i = 0; i < b.slots.size(); ++i) {
    os << "  slot " << i << ":";
    if (b.slots[i].empty()) {
      os << " []";
    }
    for (const Config &c : b.slots[i]) {
      os << " {src=";
      print_resource(c.src);
      os << " dst=[";
      if (c.dst) {
        bool first = true;
        for (mlir::Attribute attr : c.dst) {
          if (!first) {
            os << ",";
          }
          first = false;
          print_resource(mlir::dyn_cast<ResourceAttr>(attr));
        }
      }
      os << "]";
      if (c.sr.has_value()) {
        os << " sr=" << (*c.sr == 0 ? "send" : "recv");
      }
      os << "}";
    }
    os << "\n";
  }
  os << "  sequence: [";
  for (std::size_t i = 0; i < b.sequence.size(); ++i) {
    if (i) {
      os << ",";
    }
    os << b.sequence[i];
  }
  os << "]\n";
}

// Stash `current` into a binding slot (existing match preferred, else first
// empty slot), record the slot index in `binding.sequence`, and return it.
// Returns nullopt if no free slot is available.
static std::optional<std::size_t>
save_current(InterconnectBinding &binding, const std::vector<Config> &current) {
  std::optional<std::size_t> existing_slot;
  std::optional<std::size_t> free_slot;
  for (std::size_t i = 0; i < binding.slots.size(); ++i) {
    if (binding.slots[i] == current && !existing_slot.has_value()) {
      existing_slot = i;
    }
    if (binding.slots[i].empty() && !free_slot.has_value()) {
      free_slot = i;
    }
  }
  std::size_t slot_idx;
  if (existing_slot.has_value()) {
    slot_idx = *existing_slot;
  } else if (!free_slot.has_value()) {
    return std::nullopt;
  } else {
    binding.slots[*free_slot] = current;
    slot_idx = *free_slot;
  }
  binding.sequence.push_back(static_cast<int>(slot_idx));
  return slot_idx;
}

static void log_node(const Node &n, llvm::raw_ostream &os) {
  os << "id=" << n.id
     << " kind=" << (n.kind == NodeKind::First ? "first" : "last") << "\n";
}

// Drop edges from `parent` to all its children; any child that now has no
// remaining incoming edges joins `free_set`. When a child is freed, any of its
// bidirectional partners with no other edges are co-freed and inherit copies
// of the child's outgoing edges so their later pop releases the same
// downstream nodes.
static void release_children(RoutingDepGraph &graph, NodeKey parent,
                             std::vector<Node> &free_set) {
  for (const Node &child : graph.children_of(parent)) {
    graph.remove_edge(parent, child.key());
    if (!graph.has_incoming(child)) {
      free_set.push_back(child);
      for (const Node &neighbor : graph.children_of(child.key())) {
        bool is_bidir = false;
        for (const Node &nc : graph.children_of(neighbor.key())) {
          if (nc.key() == child.key()) {
            is_bidir = true;
            break;
          }
        }
        if (!is_bidir) {
          continue;
        }
        bool no_other_outgoing = true;
        for (const Node &nc : graph.children_of(neighbor.key())) {
          if (nc.key() != child.key()) {
            no_other_outgoing = false;
            break;
          }
        }
        if (graph.has_incoming(neighbor) || !no_other_outgoing) {
          continue;
        }
        graph.remove_edge(child.key(), neighbor.key());
        graph.remove_edge(neighbor.key(), child.key());
        for (const Node &gc : graph.children_of(child.key())) {
          graph.insert_edge(neighbor.key(), gc.key());
        }
        free_set.push_back(neighbor);
      }
    }
  }
}

bool has_conflict(const Node &candidate, const std::vector<Config> &current,
                  llvm::StringRef kind) {
  if (kind == "bulk") {
    int candidate_sr = (candidate.dir == "send") ? 0 : 1;
    for (const Config &c : current) {
      if (c.sr.has_value() && *c.sr == candidate_sr) {
        return true;
      }
    }
    return false;
  }
  for (const Config &c : current) {
    if (c.src == candidate.src || c.dst == candidate.dst) {
      return true;
    }
  }
  return false;
}

InterconnectBinding bind_interconnect(RoutingDepGraph graph,
                                      llvm::StringRef kind) {
  std::vector<Node> free_set;
  std::vector<Config> active;
  std::vector<Config> current;
  InterconnectBinding binding{};

  // Begin algorithm at start node
  release_children(graph, NodeKey{0, NodeKind::First}, free_set);

  // actual algorithm
  while (!free_set.empty()) {
    // Priority: non-sentinel Last > First > end sentinel.
    std::size_t pick = free_set.size();
    for (std::size_t i = 0; i < free_set.size(); ++i) {
      if (free_set[i].kind == NodeKind::Last && free_set[i].id != 0) {
        pick = i;
        break;
      }
    }
    if (pick == free_set.size()) {
      for (std::size_t i = 0; i < free_set.size(); ++i) {
        if (free_set[i].kind == NodeKind::First) {
          pick = i;
          break;
        }
      }
    }
    if (pick == free_set.size()) {
      pick = 0;
    }
    Node candidate = free_set[pick];
    free_set.erase(free_set.begin() + pick);
    log_node(candidate, llvm::errs());
    release_children(graph, candidate.key(), free_set);
    if (candidate.kind == NodeKind::First) {
      Config cfg{candidate.src, candidate.dst, std::nullopt};
      if (kind == "bulk") {
        cfg.sr = (candidate.dir == "send") ? 0 : 1;
      }
      // check if the current route is already active in the current
      // configuration
      bool already_in_current = false;
      for (const Config &c : current) {
        if (c == cfg) {
          already_in_current = true;
          break;
        }
      }
      if (already_in_current) {
        // if not in active it should be added
        bool already_in_active = false;
        for (const Config &c : active) {
          if (c == cfg) {
            already_in_active = true;
            break;
          }
        }
        if (!already_in_active) {
          active.push_back(cfg);
        }
        continue;
      }
      // if there is a conflict with the current configuration
      if (has_conflict(candidate, current, kind)) {
        if (!save_current(binding, current).has_value()) {
          llvm::errs() << "bind_interconnect: no free binding slot\n";
          return binding;
        }
        current = active;
      }
      current.push_back(cfg);
      active.push_back(cfg);
    } else {
      // when end node encountered remove the route from the active ones
      Config cfg{candidate.src, candidate.dst, std::nullopt};
      if (kind == "bulk") {
        cfg.sr = (candidate.dir == "send") ? 0 : 1;
      }
      for (auto it = active.begin(); it != active.end(); ++it) {
        if (*it == cfg) {
          active.erase(it);
          break;
        }
      }
    }
  }

  // Flush any remaining `current` into the binding.
  if (!current.empty()) {
    if (!save_current(binding, current).has_value()) {
      llvm::errs() << "bind_interconnect: no free binding slot\n";
      return binding;
    }
  }

  return binding;
}

} // namespace vesyla::pasm
