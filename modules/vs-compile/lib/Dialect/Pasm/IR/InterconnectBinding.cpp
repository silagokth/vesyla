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

// The trip count of each loop a transfer releases its route across,
// outermost first. Empty for a transfer that runs once.
static std::vector<uint32_t> own_trips(const Anchor &a) {
  std::vector<uint32_t> own;
  for (std::size_t k = 0; k < a.pass_hi.size() && k < a.ir_idx.size(); ++k) {
    own.push_back(a.pass_hi[k] - a.ir_idx[k] + 1);
  }
  return own;
}

// How often the repeated part of the binding's sequence repeats: the trip
// count of each loop its transfers release their routes across, outermost
// first. Empty when the sequence covers the whole epoch. Its transfers have to
// agree -- one that repeats over other loops would need a sequence of its own,
// which is reported on `report` when there is one. Transfers that run once
// before or after the loops are the prefix and suffix, not counted here.
static mlir::FailureOr<std::vector<uint32_t>>
pass_trips(const InterconnectBinding &binding, mlir::Operation *report) {
  std::vector<uint32_t> trips;
  bool seeded = false;
  std::size_t end = binding.sequence.size() - binding.suffix_steps;
  for (std::size_t j = binding.prefix_steps; j < end; ++j) {
    for (const InterconnectConfigOption &opt :
         binding.slots[binding.sequence[j]]) {
      for (const Anchor &a : opt.first_anchors) {
        std::vector<uint32_t> own = own_trips(a);
        if (!seeded) {
          trips = own;
          seeded = true;
        } else if (own != trips) {
          if (report) {
            report->emitError("interconnect: transfers sharing a route repeat "
                              "over different loops; not supported");
          }
          return mlir::failure();
        }
      }
    }
  }
  return trips;
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

// The part of the sequence a configuration belongs to, in the order they run.
enum Part : int { Prefix = 0, Repeated = 1, Suffix = 2 };

// Stash `current` into a binding slot (existing match preferred, else first
// empty slot), record the slot index in `binding.sequence`, and return it.
// Returns nullopt if no free slot is available.
//
// A slot is only reused within the part of the sequence (`part`, see Part) that
// filled it: a transfer of the repeated part keeps its anchors on its own
// option, and a prefix that happens to configure the same routes is switched
// in by an evt of its own anyway.
static std::optional<std::size_t>
save_current(InterconnectBinding &binding, std::array<int, 4> &slot_part,
             int part, const std::vector<InterconnectConfig> &current,
             const std::vector<Anchor> &first_anchors,
             const std::vector<Anchor> &last_anchors) {
  std::optional<std::size_t> existing_slot;
  std::optional<std::size_t> free_slot;
  for (std::size_t i = 0; i < binding.slots.size(); ++i) {
    if (binding.slots[i].size() == 1 && slot_part[i] == part &&
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
    slot_part[*free_slot] = part;
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

// The error for a binding that ran out of configuration options.
static mlir::FailureOr<InterconnectBinding>
no_free_option(mlir::Operation *report, llvm::StringRef kind,
               const InterconnectBinding &binding) {
  report->emitError("interconnect: the ")
      << kind << " transfers need more than the " << binding.slots.size()
      << " configuration options the interconnect has; not supported";
  return mlir::failure();
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

mlir::FailureOr<InterconnectBinding>
bind_interconnect(RoutingDepGraph graph, llvm::StringRef kind,
                  mlir::Operation *report) {
  std::vector<Node> free_set;
  std::vector<ActiveConfig> active;
  std::vector<InterconnectConfig> current;
  std::vector<Anchor> current_first_anchors;
  std::vector<Anchor> current_last_anchors;
  InterconnectBinding binding{};
  std::array<int, 4> slot_part{-1, -1, -1, -1};

  // The loops the deepest transfers repeat over. A transfer outside all of
  // them runs once: before the first repeated transfer it is the prefix, after
  // them the suffix.
  std::vector<uint32_t> deepest;
  for (const Node &n : graph) {
    if (n.id != 0 && n.kind == NodeKind::First &&
        own_trips(n.anchor).size() > deepest.size()) {
      deepest = own_trips(n.anchor);
    }
  }
  int part = Prefix;
  auto save = [&]() {
    if (!save_current(binding, slot_part, part, current, current_first_anchors,
                      current_last_anchors)
             .has_value()) {
      return false;
    }
    if (part == Prefix) {
      ++binding.prefix_steps;
    } else if (part == Suffix) {
      ++binding.suffix_steps;
    }
    return true;
  };

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
      // Entering the next part of the sequence closes the option of the one
      // before: a prefix route left in the repeated option would also receive
      // every pass's transfers (and the other way round).
      int want = Repeated;
      if (!deepest.empty() && own_trips(candidate.anchor).empty()) {
        want = part == Prefix ? Prefix : Suffix;
      }
      if (want != part) {
        if (want < part) {
          report->emitError("interconnect: the ")
              << kind
              << " transfers repeated over a loop are interleaved with ones "
                 "that run once; not supported";
          return mlir::failure();
        }
        if (!active.empty()) {
          report->emitError("interconnect: a ")
              << kind
              << " transfer that runs once is still in flight when the "
                 "repeated ones start or end; not supported";
          return mlir::failure();
        }
        if (!current.empty()) {
          if (!save()) {
            return no_free_option(report, kind, binding);
          }
          current.clear();
          current_first_anchors.clear();
          current_last_anchors.clear();
        }
        part = want;
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
        if (!save()) {
          return no_free_option(report, kind, binding);
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
    if (!save()) {
      return no_free_option(report, kind, binding);
    }
  }
  // Only a prefix: nothing repeats, so it is the whole sequence.
  if (binding.prefix_steps == binding.sequence.size()) {
    binding.prefix_steps = 0;
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

// One instruction of the interconnect rop, at the builder's insertion point.
static void add_instr(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef type,
                      llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>>
                          params) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  for (auto [name, value] : params) {
    attrs.push_back(builder.getNamedAttr(name, value));
  }
  InstrOp::create(
      builder, loc,
      builder.getStringAttr(vesyla::util::Common::gen_random_string(8)),
      builder.getStringAttr(type), builder.getDictionaryAttr(attrs));
}

// The rep stepping an evt through `steps` (options, in the order they are
// switched in), when there is more than one. One rep steps by a fixed stride;
// a sequence that changes stride (0, 1, 0) would need one rep per run.
static mlir::LogicalResult emit_stride(llvm::ArrayRef<int> steps, RopOp rop,
                                       mlir::OpBuilder &builder,
                                       llvm::StringRef delay, int32_t port) {
  if (steps.size() <= 1) {
    return mlir::success();
  }
  int stride = steps[1] - steps[0];
  for (std::size_t i = 2; i < steps.size(); ++i) {
    if (steps[i] - steps[i - 1] != stride) {
      return rop->emitError("interconnect: the route option sequence of this "
                            "epoch is not one fixed stride; not supported");
    }
  }
  add_instr(builder, rop.getLoc(), "rep",
            {{"delay", builder.getStringAttr(delay)},
             {"iter", builder.getI32IntegerAttr(
                          static_cast<int32_t>(steps.size()))},
             {"step", builder.getI32IntegerAttr(stride)},
             {"port", builder.getI32IntegerAttr(port)}});
  return mlir::success();
}

mlir::LogicalResult
emit_sequence_instructions(const InterconnectBinding &binding, RopOp rop,
                           mlir::OpBuilder &builder) {
  builder.setInsertionPoint(rop.getBody().front().getTerminator());

  std::size_t n = binding.sequence.size();
  std::size_t prefix = binding.prefix_steps;
  std::size_t suffix = binding.suffix_steps;
  bool parts = prefix + suffix > 0;
  if (n <= 1 && !parts) {
    return mlir::success();
  }
  mlir::FailureOr<std::vector<uint32_t>> trips = pass_trips(binding, rop);
  if (mlir::failed(trips)) {
    return mlir::failure();
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

  // The parts in the order they run: [begin, end) of the sequence, and whether
  // it repeats with every pass.
  struct Span {
    std::size_t begin, end;
    bool repeated;
    std::string delay;
  };
  llvm::SmallVector<Span> spans;
  if (prefix > 0) {
    spans.push_back({0, prefix, false, delay_name + "_prefix"});
  }
  spans.push_back({prefix, n - suffix, true, delay_name});
  if (suffix > 0) {
    spans.push_back({n - suffix, n, false, delay_name + "_suffix"});
  }

  // Each part is an evt of its own, starting at its first option; the first
  // one is the evt the rop already holds, which starts at option 0. A
  // transition follows every evt after the first and joins it to the ones
  // before (see CoalesceRopsPass: the rop reads its instructions as a postfix
  // stack).
  for (std::size_t s = 0; s < spans.size(); ++s) {
    const Span &span = spans[s];
    llvm::ArrayRef<int> steps(binding.sequence.data() + span.begin,
                              span.end - span.begin);
    if (s > 0) {
      add_instr(builder, loc, "evt",
                {{"port", builder.getI32IntegerAttr(port)},
                 {"init_addr", builder.getI32IntegerAttr(steps.front())}});
    } else if (parts && steps.front() != 0) {
      return rop->emitError("interconnect: the first route option is not "
                            "option 0");
    }
    if (mlir::failed(emit_stride(steps, rop, builder, span.delay, port))) {
      return mlir::failure();
    }

    // One level per pass loop, innermost first, replaying the repeated part.
    // Each has a delay of its own: the route fires fewer events per pass than
    // the rops it serves, so it cannot share their period. A part of one
    // option stays configured across every pass and needs none.
    if (span.repeated && steps.size() > 1) {
      for (std::size_t k = trips->size(); k-- > 0;) {
        add_instr(builder, loc, "rep",
                  {{"delay", builder.getStringAttr(delay_name + "_" +
                                                   std::to_string(k))},
                   {"iter", builder.getI32IntegerAttr(
                                static_cast<int32_t>((*trips)[k]))},
                   {"step", builder.getI32IntegerAttr(0)},
                   {"port", builder.getI32IntegerAttr(port)}});
      }
    }
    if (s > 0) {
      add_instr(builder, loc, "trans",
                {{"delay", builder.getStringAttr(delay_name + "_trans" +
                                                 std::to_string(s))},
                 {"port", builder.getI32IntegerAttr(port)}});
    }
  }
  return mlir::success();
}

// What the debug dumps show against a constraint: the pass that emitted it and
// the rule that fired.
static mlir::ArrayAttr debug_attr(mlir::OpBuilder &builder,
                                  llvm::StringRef rule) {
  return builder.getStrArrayAttr({"InterconnectPass", rule});
}

// The route constraints of a binding in parts (see InterconnectBinding): the
// prefix and suffix run once, each an evt of its own on `rop`, and the part in
// between repeats with every pass. The rules are the ones of a binding in one
// part -- an option is switched in before the first transfer that uses it, and
// the transfers of the option before are done before the switch -- with the
// rop's anchors pointing at the evt of the step's part (its MT index) and the
// transfers of a part that runs once tied to the first or last pass of the
// repeated part next to it.
static void emit_part_constraints(const InterconnectBinding &binding,
                                  RopOp rop, mlir::OpBuilder &builder,
                                  llvm::ArrayRef<uint32_t> trips,
                                  const llvm::StringSet<> &config_rops) {
  mlir::Location loc = rop.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  auto rop_ref = mlir::FlatSymbolRefAttr::get(ctx, rop.getSymName());
  auto delay = DelayAttr::get(ctx, 1, std::nullopt);

  std::size_t n = binding.sequence.size();
  std::size_t prefix = binding.prefix_steps;
  std::size_t suffix = binding.suffix_steps;
  std::size_t rep_begin = prefix;
  std::size_t rep_end = n - suffix;

  std::vector<uint32_t> first_pass_v(trips.size(), 0);
  std::vector<uint32_t> last_pass_v;
  for (uint32_t trip : trips) {
    last_pass_v.push_back(trip - 1);
  }
  llvm::ArrayRef<uint32_t> first_pass(first_pass_v);
  llvm::ArrayRef<uint32_t> last_pass(last_pass_v);
  llvm::ArrayRef<uint32_t> none;

  auto repeated = [&](std::size_t j) { return j >= rep_begin && j < rep_end; };
  // The evt step `j` belongs to, and its first step.
  auto event_of = [&](std::size_t j) -> uint32_t {
    if (j < rep_begin) {
      return 0;
    }
    uint32_t base = prefix > 0 ? 1 : 0;
    return repeated(j) ? base : base + 1;
  };
  auto begin_of = [&](std::size_t j) {
    return j < rep_begin ? 0 : (repeated(j) ? rep_begin : rep_end);
  };
  auto length_of = [&](std::size_t j) {
    return j < rep_begin ? prefix : (repeated(j) ? rep_end - rep_begin : suffix);
  };
  // Whether the rop repeats step `j`'s evt over the passes: only a repeated
  // part that switches between options does (emit_sequence_instructions).
  auto has_passes = [&](std::size_t j) {
    return repeated(j) && length_of(j) > 1;
  };

  auto range = [&](mlir::FlatSymbolRefAttr id, llvm::ArrayRef<uint32_t> or_idx,
                   uint32_t mt, llvm::ArrayRef<uint32_t> base,
                   llvm::ArrayRef<uint32_t> from, llvm::ArrayRef<uint32_t> to,
                   llvm::ArrayRef<uint32_t> tail) {
    std::vector<uint32_t> lo;
    std::vector<uint32_t> hi;
    for (std::size_t k = 0; k < from.size(); ++k) {
      lo.push_back(base[k] + from[k]);
      hi.push_back(base[k] + to[k]);
    }
    lo.insert(lo.end(), tail.begin(), tail.end());
    hi.insert(hi.end(), tail.begin(), tail.end());
    return AnchorRangeAttr::get(ctx, id, or_idx, mt, lo, or_idx, mt, hi);
  };
  // Step `j` of the rop over passes `from`..`to` (ignored for an evt that does
  // not repeat over them).
  auto step_at = [&](std::size_t j, llvm::ArrayRef<uint32_t> from,
                     llvm::ArrayRef<uint32_t> to) {
    std::vector<uint32_t> tail;
    if (length_of(j) > 1) {
      tail.push_back(static_cast<uint32_t>(j - begin_of(j)));
    }
    if (!has_passes(j)) {
      return range(rop_ref, {}, event_of(j), none, none, none, tail);
    }
    return range(rop_ref, {}, event_of(j), first_pass, from, to, tail);
  };
  // A transfer over passes `from`..`to` of the repeated loops; a transfer that
  // runs once has none.
  auto use_at = [&](const Anchor &a, llvm::ArrayRef<uint32_t> from,
                    llvm::ArrayRef<uint32_t> to) {
    llvm::ArrayRef<uint32_t> ir(a.ir_idx);
    if (own_trips(a).empty()) {
      return range(a.instr_id, a.or_idx, a.mt, none, none, none, ir);
    }
    return range(a.instr_id, a.or_idx, a.mt, ir, from, to,
                 ir.drop_front(from.size()));
  };
  auto add = [&](AnchorRangeAttr src, AnchorRangeAttr dst, llvm::StringRef rule) {
    CstrOp::create(builder, loc, src, dst, delay, builder.getBoolAttr(false))
        ->setAttr("debug", debug_attr(builder, rule));
  };

  // config -> first_use: the step that switches an option in comes before the
  // first transfer that uses it -- on every pass when the step repeats, and
  // before the first pass when its option stays on through all of them.
  for (std::size_t j = 0; j < n; ++j) {
    std::size_t slot = static_cast<std::size_t>(binding.sequence[j]);
    bool first = true;
    for (std::size_t k = 0; k < j; ++k) {
      first &= binding.sequence[k] != binding.sequence[j];
    }
    if (!first || binding.slots[slot].empty()) {
      continue;
    }
    for (const Anchor &a : binding.slots[slot][0].first_anchors) {
      if (a.instr_id && config_rops.contains(a.instr_id.getValue())) {
        continue;
      }
      if (has_passes(j)) {
        add(step_at(j, first_pass, last_pass), use_at(a, first_pass, last_pass),
            "route-first-use");
      } else {
        add(step_at(j, none, none), use_at(a, first_pass, first_pass),
            "route-first-use");
      }
    }
  }

  // last_use -> next config: the transfers of an option are done before the
  // step that switches the next one in. Out of the repeated part that is its
  // last pass, into it its first.
  for (std::size_t j = 1; j < n; ++j) {
    int prev_slot = binding.sequence[j - 1];
    int cur_slot = binding.sequence[j];
    if (prev_slot == cur_slot || binding.slots[prev_slot].empty()) {
      continue;
    }
    // A transfer carried into the next option spans the switch; see the rule
    // of the same name in emit_interconnect_constraints. Matched on the whole
    // anchor, not the instruction: coalesced transfers share one (ger's rows
    // 0..2 and its rows of A are all events of vector_load_0).
    auto carried = [&](const Anchor &a) {
      return !binding.slots[cur_slot].empty() &&
             llvm::is_contained(binding.slots[cur_slot][0].first_anchors, a);
    };
    bool from_rep = repeated(j - 1);
    bool to_rep = repeated(j);
    llvm::ArrayRef<uint32_t> src_from = !from_rep ? none
                                        : to_rep  ? first_pass
                                                  : last_pass;
    llvm::ArrayRef<uint32_t> src_to = !from_rep ? none : last_pass;
    llvm::ArrayRef<uint32_t> dst_from = !to_rep ? none : first_pass;
    llvm::ArrayRef<uint32_t> dst_to = !to_rep ? none
                                      : from_rep ? last_pass
                                                 : first_pass;
    for (const Anchor &a : binding.slots[prev_slot][0].last_anchors) {
      if (carried(a)) {
        continue;
      }
      add(use_at(a, src_from, src_to), step_at(j, dst_from, dst_to),
          "route-last-use");
    }
  }

  // The repeated part starts over on the next pass, so its last option's last
  // use is done before that pass's first step; one constraint per pass loop,
  // as in emit_interconnect_constraints.
  if (rep_end - rep_begin < 2) {
    return;
  }
  int final_slot = binding.sequence[rep_end - 1];
  if (binding.slots[final_slot].empty()) {
    return;
  }
  for (std::size_t k = 0; k < trips.size(); ++k) {
    if (trips[k] < 2) {
      continue;
    }
    std::vector<uint32_t> src_from = first_pass_v;
    std::vector<uint32_t> src_to = last_pass_v;
    std::vector<uint32_t> dst_from = first_pass_v;
    std::vector<uint32_t> dst_to = last_pass_v;
    src_to[k] = last_pass[k] - 1;
    dst_from[k] = 1;
    for (std::size_t inner = k + 1; inner < trips.size(); ++inner) {
      src_from[inner] = last_pass[inner];
      dst_to[inner] = 0;
    }
    for (const Anchor &a : binding.slots[final_slot][0].last_anchors) {
      add(use_at(a, src_from, src_to), step_at(rep_begin, dst_from, dst_to),
          "route-next-pass");
    }
  }
}

void emit_interconnect_constraints(const InterconnectBinding &binding,
                                   RopOp rop, mlir::OpBuilder &builder) {
  builder.setInsertionPointAfter(rop);
  mlir::Location loc = rop.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  auto rop_ref = mlir::FlatSymbolRefAttr::get(ctx, rop.getSymName());
  auto delay = DelayAttr::get(ctx, 1, std::nullopt);
  bool has_sequence = binding.sequence.size() >= 2;

  // The passes the sequence repeats over; none without a sequence, the route
  // rop having no repetition to index then.
  std::vector<uint32_t> trips;
  if (has_sequence || binding.prefix_steps + binding.suffix_steps > 0) {
    trips = *pass_trips(binding, /*report=*/nullptr);
  }
  std::vector<uint32_t> first_pass(trips.size(), 0);
  std::vector<uint32_t> last_pass;
  for (uint32_t trip : trips) {
    last_pass.push_back(trip - 1);
  }

  // An anchor range over passes `from`..`to` of each pass loop, counted from
  // `base`, followed by `tail` as a point. With no pass loops it is the point
  // `tail`.
  auto spanning = [&](mlir::FlatSymbolRefAttr id,
                      llvm::ArrayRef<uint32_t> or_idx, uint32_t mt,
                      llvm::ArrayRef<uint32_t> base,
                      llvm::ArrayRef<uint32_t> from,
                      llvm::ArrayRef<uint32_t> to,
                      llvm::ArrayRef<uint32_t> tail) {
    std::vector<uint32_t> lo;
    std::vector<uint32_t> hi;
    for (std::size_t k = 0; k < from.size(); ++k) {
      lo.push_back(base[k] + from[k]);
      hi.push_back(base[k] + to[k]);
    }
    lo.insert(lo.end(), tail.begin(), tail.end());
    hi.insert(hi.end(), tail.begin(), tail.end());
    return AnchorRangeAttr::get(ctx, id, or_idx, mt, lo, or_idx, mt, hi);
  };
  // A transfer's anchor, and a step of this rop's sequence, over those passes.
  auto of_use = [&](const Anchor &a, llvm::ArrayRef<uint32_t> from,
                    llvm::ArrayRef<uint32_t> to) {
    llvm::ArrayRef<uint32_t> ir(a.ir_idx);
    return spanning(a.instr_id, a.or_idx, a.mt, ir, from, to,
                    ir.drop_front(from.size()));
  };
  auto of_step = [&](llvm::ArrayRef<uint32_t> step,
                     llvm::ArrayRef<uint32_t> from,
                     llvm::ArrayRef<uint32_t> to) {
    return spanning(rop_ref, {}, 0, first_pass, from, to, step);
  };

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

  if (binding.prefix_steps + binding.suffix_steps > 0) {
    emit_part_constraints(binding, rop, builder, trips, config_rops);
    return;
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
        auto src_ar = of_step(src_indices, first_pass, last_pass);
        auto dst_ar = of_use(a, first_pass, last_pass);
        CstrOp::create(builder, loc, src_ar, dst_ar, delay,
                       builder.getBoolAttr(false))
            ->setAttr("debug", debug_attr(builder, "route-first-use"));
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
        auto src_ar = of_use(a, first_pass, last_pass);
        std::vector<uint32_t> dst_idx = {static_cast<uint32_t>(j)};
        auto dst_ar = of_step(dst_idx, first_pass, last_pass);
        CstrOp::create(builder, loc, src_ar, dst_ar, delay,
                       builder.getBoolAttr(false))
            ->setAttr("debug", debug_attr(builder, "route-last-use"));
      }
    }

    // The sequence starts over on the next pass, so the last option's last
    // use has to be done before that pass's first step. One constraint per
    // pass loop: stepping it leaves the last pass of every loop inside and
    // enters their first.
    int final_slot = binding.sequence.back();
    std::vector<uint32_t> first_step = {0};
    for (std::size_t k = 0; k < trips.size(); ++k) {
      if (trips[k] < 2 || binding.slots[final_slot].empty()) {
        continue;
      }
      std::vector<uint32_t> src_from = first_pass;
      std::vector<uint32_t> src_to = last_pass;
      std::vector<uint32_t> dst_from = first_pass;
      std::vector<uint32_t> dst_to = last_pass;
      src_to[k] = last_pass[k] - 1;
      dst_from[k] = 1;
      for (std::size_t inner = k + 1; inner < trips.size(); ++inner) {
        src_from[inner] = last_pass[inner];
        dst_to[inner] = 0;
      }
      for (const Anchor &a : binding.slots[final_slot][0].last_anchors) {
        CstrOp::create(builder, loc, of_use(a, src_from, src_to),
                       of_step(first_step, dst_from, dst_to), delay,
                       builder.getBoolAttr(false))
            ->setAttr("debug", debug_attr(builder, "route-next-pass"));
      }
    }
  }
}

} // namespace vesyla::pasm
