#include "vesyla/Dialect/Pasm/IR/InterconnectBinding.hpp"

#include "vesyla/Support/Common.hpp"

#include <cassert>

namespace vesyla::pasm {

int direction_code(ResourceAttr from, ResourceAttr to) {
  int dr = to.getRow() - from.getRow();
  int dc = to.getCol() - from.getCol();
  int sr = (dr > 0) - (dr < 0);
  int sc = (dc > 0) - (dc < 0);
  return (sr + 1) * 3 + (sc + 1);
}

static void print_anchor(const Anchor &a, llvm::raw_ostream &os) {
  if (a.instr_id) {
    os << a.instr_id.getValue();
  }
  if (!a.event.empty()) {
    os << "." << a.event;
  }
  if (!a.indices.empty()) {
    os << "[";
    for (std::size_t i = 0; i < a.indices.size(); ++i) {
      if (i) {
        os << ",";
      }
      os << a.indices[i];
    }
    os << "]";
  }
  if (a.delay != 0) {
    os << "+" << a.delay;
  }
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
    os << "  option " << i << ":";
    if (b.slots[i].empty()) {
      os << " []";
    }
    for (const InterconnectConfigOption &opt : b.slots[i]) {
      for (const InterconnectConfig &c : opt.configs) {
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
      os << "\n    first: [";
      for (std::size_t j = 0; j < opt.first_anchors.size(); ++j) {
        if (j) {
          os << ", ";
        }
        print_anchor(opt.first_anchors[j], os);
      }
      os << "]\n    last: [";
      for (std::size_t j = 0; j < opt.last_anchors.size(); ++j) {
        if (j) {
          os << ", ";
        }
        print_anchor(opt.last_anchors[j], os);
      }
      os << "]";
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
save_current(InterconnectBinding &binding,
             const std::vector<InterconnectConfig> &current,
             const std::vector<Anchor> &first_anchors,
             const std::vector<Anchor> &last_anchors) {
  std::optional<std::size_t> existing_slot;
  std::optional<std::size_t> free_slot;
  for (std::size_t i = 0; i < binding.slots.size(); ++i) {
    if (binding.slots[i].size() == 1 &&
        binding.slots[i][0].configs == current && !existing_slot.has_value()) {
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
    binding.slots[*free_slot].push_back(
        InterconnectConfigOption{current, first_anchors, last_anchors});
    slot_idx = *free_slot;
  }
  binding.sequence.push_back(static_cast<int>(slot_idx));
  return slot_idx;
}

static std::string node_tag(const Node &n) {
  return std::to_string(n.id) + (n.kind == NodeKind::First ? "f" : "l");
}

static void log_pop(const Node &candidate, const std::vector<Node> &free_set,
                    const std::vector<InterconnectConfig> &active) {
  llvm::errs() << "popped: " << node_tag(candidate) << "\n";
  llvm::errs() << "  free: [";
  for (std::size_t i = 0; i < free_set.size(); ++i) {
    if (i) {
      llvm::errs() << ", ";
    }
    llvm::errs() << node_tag(free_set[i]);
  }
  llvm::errs() << "]\n";
  llvm::errs() << "  active: [";
  for (std::size_t i = 0; i < active.size(); ++i) {
    if (i) {
      llvm::errs() << ", ";
    }
    llvm::errs() << active[i].src.getSlot() << " -> ";
    if (active[i].dst) {
      bool first = true;
      for (mlir::Attribute attr : active[i].dst) {
        if (!first) {
          llvm::errs() << " ";
        }
        first = false;
        auto r = mlir::dyn_cast<ResourceAttr>(attr);
        if (r) {
          llvm::errs() << r.getSlot();
        }
      }
    }
  }
  llvm::errs() << "]\n";
}

static void add_anchor(std::vector<Anchor> &anchors, const Anchor &a) {
  for (const Anchor &x : anchors) {
    if (x == a) {
      return;
    }
  }
  anchors.push_back(a);
}

// Drop edges from `parent` to all its children; any child that now has no
// remaining incoming edges joins `free_set`. When a child is freed, any of its
// bidirectional partners with no other edges are co-freed and inherit copies
// of the child's outgoing edges so their later pop releases the same
// downstream nodes.
static bool in_free_set(const std::vector<Node> &free_set, NodeKey key) {
  for (const Node &n : free_set) {
    if (n.key() == key) {
      return true;
    }
  }
  return false;
}

static void release_children(RoutingDepGraph &graph, NodeKey parent,
                             std::vector<Node> &free_set) {
  for (const Node &child : graph.children_of(parent)) {
    graph.remove_edge(parent, child.key());
    if (!graph.has_incoming(child) && !in_free_set(free_set, child.key())) {
      free_set.push_back(child);
      for (const Node &neighbor : graph.children_of(child.key())) {
        if (!graph.is_bidir(child.key(), neighbor.key())) {
          continue;
        }
        graph.remove_edge(child.key(), neighbor.key());
        bool no_other_outgoing = graph.children_of(neighbor.key()).empty();
        if (graph.has_incoming(neighbor) || !no_other_outgoing) {
          continue;
        }
        for (const Node &gc : graph.children_of(child.key())) {
          graph.insert_edge(neighbor.key(), gc.key());
        }
        if (!in_free_set(free_set, neighbor.key())) {
          free_set.push_back(neighbor);
        }
      }
    }
  }
}

bool has_conflict(const Node &candidate,
                  const std::vector<InterconnectConfig> &current,
                  llvm::StringRef kind) {
  if (kind == "bulk") {
    int candidate_sr = (candidate.dir == "send") ? 0 : 1;
    for (const InterconnectConfig &c : current) {
      if (c.sr.has_value() && *c.sr == candidate_sr) {
        return true;
      }
    }
    return false;
  }
  for (const InterconnectConfig &c : current) {
    if (c.src == candidate.src || c.dst == candidate.dst) {
      return true;
    }
  }
  return false;
}

InterconnectBinding bind_interconnect(RoutingDepGraph graph,
                                      llvm::StringRef kind) {
  std::vector<Node> free_set;
  std::vector<InterconnectConfig> active;
  std::vector<InterconnectConfig> current;
  std::vector<Anchor> current_first_anchors;
  std::vector<Anchor> current_last_anchors;
  InterconnectBinding binding{};

  // Begin algorithm at start node
  release_children(graph, NodeKey{0, NodeKind::First}, free_set);

  // actual algorithm
  while (free_set.size() > 1 || (!free_set.empty() && free_set[0].id != 0)) {
    // Priority: Last > First.
    std::size_t pick = free_set.size();
    for (std::size_t i = 0; i < free_set.size(); ++i) {
      if (free_set[i].kind == NodeKind::Last) {
        pick = i;
        break;
      }
    }
    if (pick == free_set.size()) {
      pick = 0;
    }
    Node candidate = free_set[pick];
    free_set.erase(free_set.begin() + pick);
    release_children(graph, candidate.key(), free_set);
    log_pop(candidate, free_set, active);
    if (candidate.kind == NodeKind::First) {
      InterconnectConfig cfg{candidate.src, candidate.dst, std::nullopt};
      if (kind == "bulk") {
        cfg.sr = (candidate.dir == "send") ? 0 : 1;
      }
      // check if the current route is already active in the current
      // configuration
      bool already_in_current = false;
      for (const InterconnectConfig &c : current) {
        if (c == cfg) {
          already_in_current = true;
          break;
        }
      }
      if (already_in_current) {
        // if not in active it should be added
        bool already_in_active = false;
        for (const InterconnectConfig &c : active) {
          if (c == cfg) {
            already_in_active = true;
            break;
          }
        }
        if (!already_in_active) {
          active.push_back(cfg);
        }
        add_anchor(current_first_anchors, candidate.anchor);
        continue;
      }
      // if there is a conflict with the current configuration
      if (has_conflict(candidate, current, kind)) {
        if (!save_current(binding, current, current_first_anchors,
                          current_last_anchors)
                 .has_value()) {
          llvm::errs() << "bind_interconnect: no free binding slot\n";
          return binding;
        }
        current = active;
        current_first_anchors.clear();
        current_last_anchors.clear();
      }
      current.push_back(cfg);
      active.push_back(cfg);
      add_anchor(current_first_anchors, candidate.anchor);
    } else {
      // when end node encountered remove the route from the active ones
      InterconnectConfig cfg{candidate.src, candidate.dst, std::nullopt};
      if (kind == "bulk") {
        cfg.sr = (candidate.dir == "send") ? 0 : 1;
      }
      add_anchor(current_last_anchors, candidate.anchor);
      for (auto it = active.begin(); it != active.end(); ++it) {
        if (*it == cfg) {
          active.erase(it);
          break;
        }
      }
    }
  }

  assert(graph.edge_count() == 0 && "edges remain after binding");

  // Flush any remaining `current` into the binding.
  if (!current.empty()) {
    if (!save_current(binding, current, current_first_anchors,
                      current_last_anchors)
             .has_value()) {
      llvm::errs() << "bind_interconnect: no free binding slot\n";
      return binding;
    }
  }

  return binding;
}

static RopOp create_interconnect_rop(CellOp cell, int32_t port,
                                     mlir::OpBuilder &builder) {
  builder.setInsertionPointToEnd(&cell.getBody().front());

  std::string prefix = (port == 0) ? "swb" : "route";
  std::string sym_name = prefix + "_" + std::to_string(cell.getRow()) + "_" +
                         std::to_string(cell.getCol());

  auto rop = RopOp::create(
      builder, cell.getLoc(), builder.getStringAttr(sym_name),
      builder.getI32IntegerAttr(cell.getRow()),
      builder.getI32IntegerAttr(cell.getCol()), builder.getI32IntegerAttr(0),
      builder.getI32IntegerAttr(port),
      /*map=*/mlir::AffineMapAttr());

  mlir::Block *body = builder.createBlock(&rop.getBody());
  builder.setInsertionPointToEnd(body);
  return rop;
}

static bool binding_empty(const InterconnectBinding &binding) {
  for (const auto &slot : binding.slots) {
    if (!slot.empty()) {
      return false;
    }
  }
  return true;
}

RopOp emit_swb_instructions(const InterconnectBinding &binding, CellOp cell,
                            mlir::OpBuilder &builder) {
  if (binding_empty(binding)) {
    return nullptr;
  }
  auto rop = create_interconnect_rop(cell, 0, builder);
  mlir::Location loc = rop.getLoc();

  for (std::size_t i = 0; i < binding.slots.size(); ++i) {
    for (const InterconnectConfigOption &opt : binding.slots[i]) {
      for (const InterconnectConfig &cfg : opt.configs) {
        auto dst_res = mlir::dyn_cast<ResourceAttr>(cfg.dst[0]);
        int32_t src_slot = cfg.src.getSlot();
        int32_t dst_slot = dst_res.getSlot();

        llvm::SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            "channel", builder.getI32IntegerAttr(dst_slot)));
        attrs.push_back(builder.getNamedAttr(
            "option", builder.getI32IntegerAttr(static_cast<int32_t>(i))));
        attrs.push_back(builder.getNamedAttr(
            "source", builder.getI32IntegerAttr(src_slot)));
        attrs.push_back(builder.getNamedAttr(
            "target", builder.getI32IntegerAttr(dst_slot)));
        attrs.push_back(
            builder.getNamedAttr("variant", builder.getStringAttr("swb")));

        InstrOp::create(
            builder, loc,
            builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
            builder.getStringAttr("conf"), builder.getDictionaryAttr(attrs));
      }
    }
  }

  YieldOp::create(builder, loc);
  return rop;
}

RopOp emit_route_instructions(const InterconnectBinding &binding, CellOp cell,
                              mlir::OpBuilder &builder) {
  if (binding_empty(binding)) {
    return nullptr;
  }
  auto rop = create_interconnect_rop(cell, 1, builder);
  mlir::Location loc = rop.getLoc();

  for (std::size_t i = 0; i < binding.slots.size(); ++i) {
    for (const InterconnectConfigOption &opt : binding.slots[i]) {
      for (const InterconnectConfig &cfg : opt.configs) {
        auto dst_res = mlir::dyn_cast<ResourceAttr>(cfg.dst[0]);
        llvm::SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            "option", builder.getI32IntegerAttr(static_cast<int32_t>(i))));
        attrs.push_back(builder.getNamedAttr(
            "sr", builder.getI32IntegerAttr(cfg.sr.value_or(0))));

        if (cfg.sr.value_or(0) == 0) {
          attrs.push_back(builder.getNamedAttr(
              "source", builder.getI32IntegerAttr(cfg.src.getSlot())));
          attrs.push_back(builder.getNamedAttr(
              "target",
              builder.getI32IntegerAttr(1 << direction_code(cfg.src, dst_res))));
        } else {
          attrs.push_back(builder.getNamedAttr(
              "source",
              builder.getI32IntegerAttr(direction_code(dst_res, cfg.src))));
          int32_t target_mask = 0;
          for (mlir::Attribute attr : cfg.dst) {
            auto r = mlir::dyn_cast<ResourceAttr>(attr);
            if (r) {
              target_mask |= (1 << r.getSlot());
            }
          }
          attrs.push_back(builder.getNamedAttr(
              "target", builder.getI32IntegerAttr(target_mask)));
        }
        attrs.push_back(
            builder.getNamedAttr("variant", builder.getStringAttr("route")));

        InstrOp::create(
            builder, loc,
            builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
            builder.getStringAttr("conf"), builder.getDictionaryAttr(attrs));
      }
    }
  }

  YieldOp::create(builder, loc);
  return rop;
}

void emit_sequence_instructions(const InterconnectBinding &binding, RopOp rop,
                                mlir::OpBuilder &builder) {
  builder.setInsertionPoint(rop.getBody().front().getTerminator());

  if (binding.sequence.size() <= 1) {
    return;
  }

  std::vector<int> deltas;
  for (std::size_t i = 1; i < binding.sequence.size(); ++i) {
    deltas.push_back(binding.sequence[i] - binding.sequence[i - 1]);
  }

  llvm::errs() << "deltas: [";
  for (std::size_t i = 0; i < deltas.size(); ++i) {
    if (i) {
      llvm::errs() << ", ";
    }
    llvm::errs() << deltas[i];
  }
  llvm::errs() << "]\n";

  std::vector<std::pair<int, int>> runs;
  int current = deltas[0];
  int count = 1;
  for (std::size_t i = 1; i < deltas.size(); ++i) {
    if (deltas[i] == current) {
      ++count;
    } else {
      runs.push_back({current, count});
      current = deltas[i];
      count = 1;
    }
  }
  runs.push_back({current, count});

  if (runs.size() != 1) {
    llvm::errs() << "emit_sequence_instructions: multiple runs not implemented "
                    "yet\n";
    return;
  }

  mlir::Location loc = rop.getLoc();
  int32_t port = rop.getPort();

  llvm::SmallVector<mlir::NamedAttribute> evt_attrs;
  evt_attrs.push_back(
      builder.getNamedAttr("port", builder.getI32IntegerAttr(port)));
  InstrOp::create(
      builder, loc,
      builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
      builder.getStringAttr("evt"), builder.getDictionaryAttr(evt_attrs));

  std::string kind_str = (port == 0) ? "swb" : "route";
  std::string delay_name = "t_" + kind_str + "_" +
                           std::to_string(rop.getRow()) + "_" +
                           std::to_string(rop.getCol());

  llvm::SmallVector<mlir::NamedAttribute> rep_attrs;
  rep_attrs.push_back(
      builder.getNamedAttr("delay", builder.getStringAttr(delay_name)));
  rep_attrs.push_back(
      builder.getNamedAttr("iter",
                          builder.getI32IntegerAttr(runs[0].second + 1)));
  rep_attrs.push_back(
      builder.getNamedAttr("step", builder.getI32IntegerAttr(runs[0].first)));
  InstrOp::create(
      builder, loc,
      builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
      builder.getStringAttr("rep"), builder.getDictionaryAttr(rep_attrs));
}

void emit_interconnect_constraints(const InterconnectBinding &binding,
                                   RopOp rop, mlir::OpBuilder &builder) {
  builder.setInsertionPointAfter(rop);
  mlir::Location loc = rop.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  auto rop_ref = mlir::FlatSymbolRefAttr::get(ctx, rop.getSymName());
  auto delay = DelayAttr::get(ctx, 1, std::nullopt);
  bool has_sequence = binding.sequence.size() >= 2;

  // config -> first_use: rop sequence step must happen before first use
  for (std::size_t i = 0; i < binding.slots.size(); ++i) {
    for (const InterconnectConfigOption &opt : binding.slots[i]) {
      for (const Anchor &a : opt.first_anchors) {
        std::vector<uint32_t> src_indices;
        std::string src_event;
        if (has_sequence) {
          src_event = "e0";
          for (std::size_t j = 0; j < binding.sequence.size(); ++j) {
            if (static_cast<std::size_t>(binding.sequence[j]) == i) {
              src_indices.push_back(static_cast<uint32_t>(j));
            }
          }
        }
        auto src_ar = AnchorRangeAttr::get(ctx, rop_ref, src_event,
                                           src_indices, src_indices);
        auto dst_ar = AnchorRangeAttr::get(ctx, a.instr_id, a.event,
                                           a.indices, a.indices);
        CstrOp::create(builder, loc, src_ar, dst_ar, delay,
                       builder.getBoolAttr(false));
      }
    }
  }

  // last_use -> next config: previous config's last use must finish before
  // the next reconfiguration step
  if (has_sequence) {
    for (std::size_t j = 1; j < binding.sequence.size(); ++j) {
      int prev_slot = binding.sequence[j - 1];
      int cur_slot = binding.sequence[j];
      if (prev_slot == cur_slot) {
        continue;
      }
      const InterconnectConfigOption &prev_opt = binding.slots[prev_slot][0];
      for (const Anchor &a : prev_opt.last_anchors) {
        auto src_ar = AnchorRangeAttr::get(ctx, a.instr_id, a.event,
                                           a.indices, a.indices);
        std::vector<uint32_t> dst_idx = {static_cast<uint32_t>(j)};
        auto dst_ar =
            AnchorRangeAttr::get(ctx, rop_ref, "e0", dst_idx, dst_idx);
        CstrOp::create(builder, loc, src_ar, dst_ar, delay,
                       builder.getBoolAttr(false));
      }
    }
  }
}

} // namespace vesyla::pasm
