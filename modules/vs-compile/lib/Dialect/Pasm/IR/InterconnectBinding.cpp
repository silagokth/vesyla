#include "vesyla/Dialect/Pasm/IR/InterconnectBinding.hpp"

#include "vesyla/Support/Anchor.hpp"
#include "vesyla/Support/Common.hpp"

#include "llvm/ADT/StringSet.h"

#include <cassert>

namespace vesyla::pasm {

int direction_code(ResourceAttr from, ResourceAttr to) {
  int dr = to.getRow() - from.getRow();
  int dc = to.getCol() - from.getCol();
  int sr = (dr > 0) - (dr < 0);
  int sc = (dc > 0) - (dc < 0);
  return (sr + 1) * 3 + (sc + 1);
}

ConfigKey config_key(const InterconnectConfig &cfg) {
  auto dst_res = cfg.dst && !cfg.dst.empty()
                     ? mlir::dyn_cast<ResourceAttr>(cfg.dst[0])
                     : ResourceAttr();
  if (!cfg.sr.has_value()) {
    // An swb config names both ends by slot; the channel it is written on is
    // the target slot, so the pair is the whole identity.
    return ConfigKey{-1, cfg.src ? cfg.src.getSlot() : -1,
                     dst_res ? dst_res.getSlot() : -1};
  }
  if (*cfg.sr == 0) {
    return ConfigKey{0, cfg.src ? cfg.src.getSlot() : -1,
                     dst_res ? 1 << direction_code(cfg.src, dst_res) : 0};
  }
  int target_mask = 0;
  for (mlir::Attribute attr : cfg.dst) {
    if (auto r = mlir::dyn_cast<ResourceAttr>(attr)) {
      target_mask |= (1 << r.getSlot());
    }
  }
  return ConfigKey{1, dst_res ? direction_code(dst_res, cfg.src) : -1,
                   target_mask};
}

bool InterconnectConfig::operator==(const InterconnectConfig &o) const {
  return config_key(*this) == config_key(o);
}

// Build a point AnchorRangeAttr (lo == hi) from OR/MT/IR indices.
static AnchorRangeAttr make_point_anchor_range(mlir::MLIRContext *ctx,
                                               mlir::FlatSymbolRefAttr id,
                                               llvm::ArrayRef<uint32_t> or_idx,
                                               uint32_t mt,
                                               llvm::ArrayRef<uint32_t> ir) {
  return AnchorRangeAttr::get(ctx, id, or_idx, mt, ir, or_idx, mt, ir);
}

static void print_anchor(const Anchor &a, llvm::raw_ostream &os) {
  if (a.instr_id) {
    os << a.instr_id.getValue();
  }
  ::vesyla::Anchor va;
  va.or_idx.assign(a.or_idx.begin(), a.or_idx.end());
  va.mt_idx = static_cast<int>(a.mt);
  va.ir_idx.assign(a.ir_idx.begin(), a.ir_idx.end());
  os << va.to_string(); // empty name -> just the OR.MT.IR index text
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

// Whether two option sets hold the same configs. A set rather than a sequence:
// the walk builds an option in whatever order the transfers came free, and two
// options that configure the fabric identically should share a slot rather
// than take one each.
static bool same_configs(const std::vector<InterconnectConfig> &a,
                         const std::vector<InterconnectConfig> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  std::vector<bool> taken(b.size(), false);
  for (const InterconnectConfig &x : a) {
    bool found = false;
    for (std::size_t i = 0; i < b.size(); ++i) {
      if (!taken[i] && b[i] == x) {
        taken[i] = true;
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
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
        same_configs(binding.slots[i][0].configs, current) &&
        !existing_slot.has_value()) {
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

// A route in flight, and how many transfers still need it. Several transfers
// share one configuration once identity is what the fabric sees, so the first
// of them to finish must not retire a route the others are still using.
struct ActiveConfig {
  InterconnectConfig config;
  int uses;
};

// Whether `held` already carries what `cfg` asks for. The same thing as
// equality everywhere but a recv, which names a set of slots: one that listens
// on a direction for slots 2 and 3 is also the config a transfer into slot 3
// needs, so it is taken up rather than opened beside it.
static bool config_covers(const InterconnectConfig &held,
                          const InterconnectConfig &cfg) {
  ConfigKey h = config_key(held);
  ConfigKey c = config_key(cfg);
  if (h.sr != c.sr || h.source != c.source) {
    return false;
  }
  if (h.sr != 1) {
    return h.target == c.target;
  }
  return (h.target & c.target) == c.target;
}

// The route another transfer already has in flight, or null.
static ActiveConfig *find_active(std::vector<ActiveConfig> &active,
                                 const InterconnectConfig &cfg) {
  for (ActiveConfig &a : active) {
    if (config_covers(a.config, cfg)) {
      return &a;
    }
  }
  return nullptr;
}

// Add a transfer to a route in flight, opening it if it is not already.
static void take_up(std::vector<ActiveConfig> &active,
                    const InterconnectConfig &cfg) {
  if (ActiveConfig *held = find_active(active, cfg)) {
    held->uses++;
    return;
  }
  active.push_back(ActiveConfig{cfg, 1});
}

static bool holds_config(const std::vector<InterconnectConfig> &configs,
                         const InterconnectConfig &cfg) {
  for (const InterconnectConfig &c : configs) {
    if (config_covers(c, cfg)) {
      return true;
    }
  }
  return false;
}

// The destination slots of both, without repeats.
static mlir::ArrayAttr union_dst(mlir::ArrayAttr a, mlir::ArrayAttr b) {
  llvm::SmallVector<mlir::Attribute> merged(a.begin(), a.end());
  for (mlir::Attribute attr : b) {
    auto r = mlir::dyn_cast<ResourceAttr>(attr);
    if (!r) {
      continue;
    }
    bool present = false;
    for (mlir::Attribute existing : merged) {
      auto e = mlir::dyn_cast<ResourceAttr>(existing);
      if (e && e.getSlot() == r.getSlot()) {
        present = true;
        break;
      }
    }
    if (!present) {
      merged.push_back(attr);
    }
  }
  return mlir::ArrayAttr::get(a.getContext(), merged);
}

// Fold a recv into the config the current option already carries for that
// direction, widening its mask to cover the new slot as well.
//
// The fabric takes one recv per direction, carrying the set of slots it feeds;
// two transfers arriving from the same neighbour are that one config with two
// slots in its mask, not two configs. Opening a second one instead makes them
// conflict, which costs a configuration option per transfer and a switch
// between them that the resource then has to be told about. What each
// receiving slot keeps of the traffic is decided by its own event, not by the
// route -- which is what makes the widened mask the same program.
//
// Only against the current option: two recvs that belong to phases that
// exclude each other still take an option each.
//
// Returns false when there is nothing to fold into -- a send, the first recv
// from that direction, or one the current config already covers, which the
// ordinary take-up path handles.
static bool merge_recv(std::vector<InterconnectConfig> &current,
                       std::vector<ActiveConfig> &active,
                       const InterconnectConfig &cfg) {
  if (!cfg.sr.has_value() || *cfg.sr != 1) {
    return false;
  }
  ConfigKey key = config_key(cfg);
  for (InterconnectConfig &held : current) {
    ConfigKey held_key = config_key(held);
    if (held_key.sr != 1 || held_key.source != key.source) {
      continue;
    }
    if ((held_key.target & key.target) == key.target) {
      return false;
    }
    InterconnectConfig widened = held;
    widened.dst = union_dst(held.dst, cfg.dst);
    bool was_active = false;
    for (ActiveConfig &a : active) {
      if (config_key(a.config) == held_key) {
        a.config = widened;
        a.uses++;
        was_active = true;
      }
    }
    if (!was_active) {
      active.push_back(ActiveConfig{widened, 1});
    }
    held = widened;
    return true;
  }
  return false;
}

static std::vector<InterconnectConfig>
active_configs(const std::vector<ActiveConfig> &active) {
  std::vector<InterconnectConfig> configs;
  configs.reserve(active.size());
  for (const ActiveConfig &a : active) {
    configs.push_back(a.config);
  }
  return configs;
}

static void log_pop(const Node &candidate, const std::vector<Node> &free_set,
                    const std::vector<ActiveConfig> &active) {
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
    llvm::errs() << active[i].config.src.getSlot() << " -> ";
    if (active[i].config.dst) {
      bool first = true;
      for (mlir::Attribute attr : active[i].config.dst) {
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
  std::vector<ActiveConfig> active;
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
      // A recv from a direction the current option already listens to widens
      // that config rather than opening one beside it.
      if (merge_recv(current, active, cfg)) {
        add_anchor(current_first_anchors, candidate.anchor);
        continue;
      }
      // A route the current configuration already carries is taken up rather
      // than opened a second time.
      if (holds_config(current, cfg)) {
        take_up(active, cfg);
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
        // What is still in flight has to stay configured across the switch.
        current = active_configs(active);
        current_first_anchors.clear();
        current_last_anchors.clear();
      }
      // The flush may have brought the route back with everything else still
      // in flight, so ask again rather than listing it twice.
      if (!holds_config(current, cfg)) {
        current.push_back(cfg);
      }
      take_up(active, cfg);
      add_anchor(current_first_anchors, candidate.anchor);
    } else {
      // when end node encountered remove the route from the active ones
      InterconnectConfig cfg{candidate.src, candidate.dst, std::nullopt};
      if (kind == "bulk") {
        cfg.sr = (candidate.dir == "send") ? 0 : 1;
      }
      add_anchor(current_last_anchors, candidate.anchor);
      // The route closes when the last transfer using it is done, not the
      // first.
      for (auto it = active.begin(); it != active.end(); ++it) {
        if (config_covers(it->config, cfg)) {
          if (--it->uses <= 0) {
            active.erase(it);
          }
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

  // The port is no longer carried on the rop. The swb/route distinction is
  // recoverable from the sym_name prefix, and the individual instructions
  // (evt/rep) carry their own port where the ISA needs it.
  auto rop = RopOp::create(
      builder, cell.getLoc(), builder.getStringAttr(sym_name),
      builder.getI32IntegerAttr(cell.getRow()),
      builder.getI32IntegerAttr(cell.getCol()), builder.getI32IntegerAttr(0),
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
        ConfigKey key = config_key(cfg);

        llvm::SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            "channel", builder.getI32IntegerAttr(key.target)));
        attrs.push_back(builder.getNamedAttr(
            "option", builder.getI32IntegerAttr(static_cast<int32_t>(i))));
        attrs.push_back(builder.getNamedAttr(
            "source", builder.getI32IntegerAttr(key.source)));
        attrs.push_back(builder.getNamedAttr(
            "target", builder.getI32IntegerAttr(key.target)));
        attrs.push_back(
            builder.getNamedAttr("variant", builder.getStringAttr("swb")));

        InstrOp::create(
            builder, loc,
            builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
            builder.getStringAttr("conf"), builder.getDictionaryAttr(attrs));
      }
    }
  }

  // The swb rop addresses port 0; carry that onto the event explicitly (the
  // rop no longer holds a port, and AddSlotPortPass no longer back-fills one).
  llvm::SmallVector<mlir::NamedAttribute> evt_attrs;
  evt_attrs.push_back(
      builder.getNamedAttr("port", builder.getI32IntegerAttr(0)));
  InstrOp::create(builder, loc,
                  builder.getStringAttr(rop.getSymName().str() + "_evt"),
                  builder.getStringAttr("evt"),
                  builder.getDictionaryAttr(evt_attrs));

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
        ConfigKey key = config_key(cfg);
        llvm::SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            "option", builder.getI32IntegerAttr(static_cast<int32_t>(i))));
        attrs.push_back(builder.getNamedAttr(
            "sr", builder.getI32IntegerAttr(cfg.sr.value_or(0))));
        attrs.push_back(builder.getNamedAttr(
            "source", builder.getI32IntegerAttr(key.source)));
        attrs.push_back(builder.getNamedAttr(
            "target", builder.getI32IntegerAttr(key.target)));
        attrs.push_back(
            builder.getNamedAttr("variant", builder.getStringAttr("route")));

        InstrOp::create(
            builder, loc,
            builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
            builder.getStringAttr("conf"), builder.getDictionaryAttr(attrs));
      }
    }
  }

  llvm::SmallVector<mlir::NamedAttribute> evt_attrs;
  evt_attrs.push_back(
      builder.getNamedAttr("port", builder.getI32IntegerAttr(1)));
  InstrOp::create(builder, loc,
                  builder.getStringAttr(rop.getSymName().str() + "_evt"),
                  builder.getStringAttr("evt"),
                  builder.getDictionaryAttr(evt_attrs));

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

  // The rop no longer carries a port; recover the swb/route distinction (and
  // the matching port index) from the sym_name prefix set in
  // create_interconnect_rop.
  bool is_swb = rop.getSymName().starts_with("swb");
  int32_t port = is_swb ? 0 : 1;
  std::string kind_str = is_swb ? "swb" : "route";
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
  rep_attrs.push_back(
      builder.getNamedAttr("port", builder.getI32IntegerAttr(port)));
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

  // Collect the rops whose instruction is a config. A config first-use is itself
  // a configuration step and must not receive a config -> first_use constraint.
  llvm::StringSet<> config_rops;
  if (auto cell = rop->getParentOfType<CellOp>()) {
    cell.walk([&](InstrOp instr) {
      if (instr.getType() == "conf") {
        if (auto r = instr->getParentOfType<RopOp>()) {
          config_rops.insert(r.getSymName());
        }
      }
    });
  }

  // config -> first_use: rop sequence step must happen before first use
  for (std::size_t i = 0; i < binding.slots.size(); ++i) {
    for (const InterconnectConfigOption &opt : binding.slots[i]) {
      for (const Anchor &a : opt.first_anchors) {
        if (a.instr_id && config_rops.contains(a.instr_id.getValue())) {
          continue;
        }
        std::vector<uint32_t> src_indices;
        if (has_sequence) {
          for (std::size_t j = 0; j < binding.sequence.size(); ++j) {
            if (static_cast<std::size_t>(binding.sequence[j]) == i) {
              src_indices.push_back(static_cast<uint32_t>(j));
            }
          }
        }
        auto src_ar =
            make_point_anchor_range(ctx, rop_ref, /*or=*/{}, /*mt=*/0,
                                    src_indices);
        auto dst_ar =
            make_point_anchor_range(ctx, a.instr_id, a.or_idx, a.mt, a.ir_idx);
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
      if (binding.slots[prev_slot].empty()) {
        continue;
      }
      const InterconnectConfigOption &prev_opt = binding.slots[prev_slot][0];

      // A transfer that uses the option being switched to cannot also be
      // asked to finish before the switch. The rule above already orders it
      // after that option, and the two together are unsatisfiable however the
      // rest of the schedule falls out. It arises wherever a route outlives a
      // reconfiguration -- what is still in flight is carried into the next
      // option -- so such a transfer genuinely spans the switch and only the
      // first ordering has anything to say about it.
      llvm::StringSet<> uses_next;
      if (!binding.slots[cur_slot].empty()) {
        for (const Anchor &a : binding.slots[cur_slot][0].first_anchors) {
          if (a.instr_id) {
            uses_next.insert(a.instr_id.getValue());
          }
        }
      }

      for (const Anchor &a : prev_opt.last_anchors) {
        if (a.instr_id && uses_next.contains(a.instr_id.getValue())) {
          continue;
        }
        auto src_ar =
            make_point_anchor_range(ctx, a.instr_id, a.or_idx, a.mt, a.ir_idx);
        std::vector<uint32_t> dst_idx = {static_cast<uint32_t>(j)};
        auto dst_ar =
            make_point_anchor_range(ctx, rop_ref, /*or=*/{}, /*mt=*/0, dst_idx);
        CstrOp::create(builder, loc, src_ar, dst_ar, delay,
                       builder.getBoolAttr(false));
      }
    }
  }
}

} // namespace vesyla::pasm
