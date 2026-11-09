#include "vesyla/Dialect/Pasm/IR/InterconnectBinding.hpp"

#include "vesyla/Support/Anchor.hpp"
#include "vesyla/Support/Common.hpp"

#include "llvm/ADT/StringSet.h"

#include <cassert>
#include <map>

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

// An anchor range from lo/hi with the slopes of `a`'s loops on the tail
// dimensions (from `n_pass` on) flagged in `tail_last`: a sloped loop's last
// index moves with the loop it follows (AGU Slope Levels). Pass dimensions
// with a slope are not supported (checked in emit_sequence_instructions).
static AnchorRangeAttr sloped_range(mlir::MLIRContext *ctx,
                                    mlir::FlatSymbolRefAttr id,
                                    llvm::ArrayRef<uint32_t> or_idx, uint32_t mt,
                                    llvm::ArrayRef<uint32_t> lo,
                                    llvm::ArrayRef<uint32_t> hi,
                                    const Anchor *a, std::size_t n_pass,
                                    llvm::ArrayRef<uint8_t> tail_last) {
  bool sloped = false;
  llvm::SmallVector<int32_t> lo_slope(lo.size(), 0), hi_slope(lo.size(), 0);
  llvm::SmallVector<uint32_t> follows(lo.size(), 0);
  if (a) {
    for (std::size_t k = n_pass; k < lo.size() && k < a->loop_slope.size(); ++k) {
      bool last = k - n_pass < tail_last.size() && tail_last[k - n_pass];
      if (a->loop_slope[k] != 0 && a->loop_follows[k] != 0 && last) {
        lo_slope[k] = hi_slope[k] = a->loop_slope[k];
        follows[k] = a->loop_follows[k];
        sloped = true;
      }
    }
  }
  if (!sloped) {
    return AnchorRangeAttr::get(ctx, id, or_idx, mt, lo, or_idx, mt, hi);
  }
  return AnchorRangeAttr::get(ctx, id, or_idx, mt, lo, or_idx, mt, hi, lo_slope,
                              hi_slope, follows);
}

// Whether a transfer's loop at depth `d` is sloped.
static bool loop_sloped(const Anchor &a, std::size_t d) {
  return d < a.loop_slope.size() && a.loop_slope[d] != 0 && a.loop_follows[d] != 0;
}

// Same loop nest: both run once, or one's loops enclose the other's. Sibling
// nests (a main loop and its remainder) are not.
static bool same_nest(llvm::ArrayRef<mlir::Attribute> a,
                      llvm::ArrayRef<mlir::Attribute> b) {
  if (a.empty() || b.empty()) {
    return a.empty() == b.empty();
  }
  std::size_t n = std::min(a.size(), b.size());
  return a.take_front(n) == b.take_front(n);
}

// How often the sequence of a binding in one part repeats: the trip count of
// each loop its transfers release their routes across, outermost first. Its
// transfers have to agree -- one that repeats over other loops would need a
// part of its own, which is reported on `report` when there is one.
static mlir::FailureOr<std::vector<uint32_t>>
pass_trips(const InterconnectBinding &binding, mlir::Operation *report) {
  std::vector<uint32_t> trips;
  bool seeded = false;
  for (std::size_t j = 0; j < binding.sequence.size(); ++j) {
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

// Whether loops `a` lead `b`: a transfer over `a` runs in the nest of `b`.
static bool leads(llvm::ArrayRef<mlir::Attribute> a,
                  llvm::ArrayRef<mlir::Attribute> b) {
  return a.size() <= b.size() && a == b.take_front(a.size());
}

// A part that switches options replays them per pass of its loops. A transfer
// over fewer of them runs before or after the deeper loops, so in one pass of
// them (pinned_passes).
static mlir::LogicalResult check_part_trips(const InterconnectBinding &binding,
                                            const BindingPart &part,
                                            mlir::Operation *report) {
  if (part.end - part.begin < 2) {
    return mlir::success();
  }
  for (std::size_t j = part.begin; j < part.end; ++j) {
    for (const Anchor &a : binding.step_first[j]) {
      if (part.siblings ? !leads(part.loops, a.loops)
                        : !leads(a.loops, part.loops)) {
        return report->emitError("interconnect: transfers sharing a route "
                                 "repeat over different loops; not supported");
      }
    }
  }
  return mlir::success();
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
  for (const BindingPart &p : b.parts) {
    os << "  part [" << p.begin << "," << p.end << ") trips (";
    for (std::size_t k = 0; k < p.trips.size(); ++k) {
      os << (k ? "," : "") << p.trips[k];
    }
    os << ")\n";
  }
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
// empty slot), record the slot index in `binding.sequence` with the step's
// transfers, and return it. Returns nullopt if no free slot is available.
//
// Slots are shared across parts; anchors are kept per step.
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
  binding.step_first.push_back(first_anchors);
  binding.step_last.push_back(last_anchors);
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

// Whether configs `a` and `b` can be one option: no swb target with two
// sources, no route direction (sr) used twice.
static bool fits_one_option(const std::vector<InterconnectConfig> &a,
                            const std::vector<InterconnectConfig> &b,
                            llvm::StringRef kind) {
  for (const InterconnectConfig &y : b) {
    if (llvm::is_contained(a, y)) {
      continue;
    }
    for (const InterconnectConfig &x : a) {
      if (kind == "bulk" ? x.sr == y.sr
                         : config_key(x).target == config_key(y).target) {
        return false;
      }
    }
  }
  return true;
}

// Renumber options in order of first use, sharing identical ones and dropping
// unused ones.
static void compact_slots(InterconnectBinding &binding) {
  std::vector<std::vector<InterconnectConfigOption>> slots(
      binding.slots.size());
  std::vector<int> to(binding.slots.size(), -1);
  std::size_t used = 0;
  for (int &s : binding.sequence) {
    if (to[s] < 0) {
      for (std::size_t i = 0; i < used; ++i) {
        if (same_configs(slots[i][0].configs, binding.slots[s][0].configs)) {
          to[s] = static_cast<int>(i);
        }
      }
      if (to[s] < 0) {
        slots[used] = binding.slots[s];
        to[s] = static_cast<int>(used++);
      }
    }
    s = to[s];
  }
  binding.slots = std::move(slots);
}

// Merge adjacent one-step parts, and adjacent steps of sibling nests, whose
// options fit one option (nothing switches between them); drop empty parts.
static void merge_parts(InterconnectBinding &binding,
                        std::vector<BindingPart> &parts,
                        llvm::StringRef kind) {
  InterconnectBinding merged;
  std::vector<BindingPart> kept;
  for (const BindingPart &p : parts) {
    if (p.begin == p.end) {
      continue;
    }
    int slot = binding.sequence[p.begin];
    if (p.end - p.begin == 1 && !kept.empty() &&
        kept.back().end - kept.back().begin == 1 &&
        fits_one_option(binding.slots[merged.sequence.back()][0].configs,
                        binding.slots[slot][0].configs, kind)) {
      InterconnectConfigOption &opt = binding.slots[merged.sequence.back()][0];
      for (const InterconnectConfig &c : binding.slots[slot][0].configs) {
        if (!llvm::is_contained(opt.configs, c)) {
          opt.configs.push_back(c);
        }
      }
      for (const Anchor &a : binding.step_first[p.begin]) {
        add_anchor(merged.step_first.back(), a);
        add_anchor(opt.first_anchors, a);
      }
      for (const Anchor &a : binding.step_last[p.begin]) {
        add_anchor(merged.step_last.back(), a);
        add_anchor(opt.last_anchors, a);
      }
      if (kept.back().trips != p.trips) {
        kept.back().trips.clear();
        kept.back().loops.clear();
      }
      continue;
    }
    kept.push_back(
        BindingPart{merged.sequence.size(), 0, p.trips, p.loops, p.siblings});
    for (std::size_t j = p.begin; j < p.end; ++j) {
      int at = binding.sequence[j];
      if (p.siblings && j > p.begin &&
          fits_one_option(binding.slots[merged.sequence.back()][0].configs,
                          binding.slots[at][0].configs, kind)) {
        InterconnectConfigOption &opt =
            binding.slots[merged.sequence.back()][0];
        for (const InterconnectConfig &c : binding.slots[at][0].configs) {
          if (!llvm::is_contained(opt.configs, c)) {
            opt.configs.push_back(c);
          }
        }
        for (const Anchor &a : binding.step_first[j]) {
          add_anchor(merged.step_first.back(), a);
          add_anchor(opt.first_anchors, a);
        }
        for (const Anchor &a : binding.step_last[j]) {
          add_anchor(merged.step_last.back(), a);
          add_anchor(opt.last_anchors, a);
        }
        continue;
      }
      merged.sequence.push_back(at);
      merged.step_first.push_back(binding.step_first[j]);
      merged.step_last.push_back(binding.step_last[j]);
    }
    kept.back().end = merged.sequence.size();
  }
  binding.sequence = std::move(merged.sequence);
  binding.step_first = std::move(merged.step_first);
  binding.step_last = std::move(merged.step_last);
  parts = std::move(kept);
  compact_slots(binding);
}

// The error for a binding that ran out of configuration options.
static mlir::FailureOr<InterconnectBinding>
no_free_option(mlir::Operation *report, llvm::StringRef kind,
               const InterconnectBinding &binding) {
  report->emitError("interconnect: the ")
      << kind << " transfers need more than the " << binding.slots.size()
      << " configuration options the swb has (NUM_OPTIONS); not supported";
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
  // The swb picks one source per target slot; one source may feed several.
  int target = config_key({candidate.src, candidate.dst, std::nullopt}).target;
  for (const InterconnectConfig &c : current) {
    if (config_key(c).target == target) {
      return true;
    }
  }
  return false;
}

mlir::FailureOr<InterconnectBinding>
bind_interconnect(RoutingDepGraph graph, llvm::StringRef kind,
                  mlir::Operation *report, std::size_t num_options) {
  std::vector<Node> free_set;
  std::vector<ActiveConfig> active;
  std::vector<InterconnectConfig> current;
  std::vector<Anchor> current_first_anchors;
  std::vector<Anchor> current_last_anchors;
  InterconnectBinding binding{};
  binding.slots.resize(num_options);

  // Parts in walk order; `part_trips` is the current part's loops.
  std::vector<BindingPart> parts;
  std::optional<std::vector<uint32_t>> part_trips;
  std::vector<mlir::Attribute> part_loops;
  auto save = [&]() {
    return save_current(binding, current, current_first_anchors,
                        current_last_anchors)
        .has_value();
  };

  // Begin algorithm at start node
  release_children(graph, NodeKey{0, NodeKind::First}, free_set);

  // actual algorithm
  while (free_set.size() > 1 || (!free_set.empty() && free_set[0].id != 0)) {
    // Priority: Last > First of the current part > any First.
    std::size_t pick = free_set.size();
    for (std::size_t i = 0; i < free_set.size(); ++i) {
      if (free_set[i].kind == NodeKind::Last) {
        pick = i;
        break;
      }
    }
    for (std::size_t i = 0; pick == free_set.size() && i < free_set.size();
         ++i) {
      if (free_set[i].id != 0 && part_trips &&
          same_nest(free_set[i].anchor.loops, part_loops)) {
        pick = i;
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
      // Another loop nest starts the next part and closes the current option.
      std::vector<uint32_t> trips = own_trips(candidate.anchor);
      bool next_part =
          part_trips && !same_nest(candidate.anchor.loops, part_loops);
      // A sibling nest under loops the part shares stays in it: the part
      // switches to the nest's option and replays per pass of those loops.
      std::size_t shared = 0;
      while (next_part && shared < part_loops.size() &&
             shared < candidate.anchor.loops.size() &&
             part_loops[shared] == candidate.anchor.loops[shared]) {
        ++shared;
      }
      if (next_part && shared > 0 && active.empty() &&
          (parts.back().siblings ||
           binding.sequence.size() == parts.back().begin)) {
        part_loops.resize(shared);
        trips.resize(shared);
        part_trips = trips;
        parts.back().loops = part_loops;
        parts.back().trips = trips;
        parts.back().siblings = true;
        next_part = false;
        // Each nest its own step; steps that fit one option merge later.
        if (!current.empty()) {
          if (!save()) {
            return no_free_option(report, kind, binding);
          }
          current.clear();
          current_first_anchors.clear();
          current_last_anchors.clear();
        }
      }
      if (next_part) {
        if (!active.empty()) {
          report->emitError("interconnect: a ")
              << kind
              << " transfer is still in flight when the loops the transfers "
                 "repeat over change; not supported";
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
        parts.back().end = binding.sequence.size();
      }
      if (!part_trips || next_part) {
        parts.push_back(
            BindingPart{binding.sequence.size(), 0, trips, candidate.anchor.loops});
        part_trips = trips;
        part_loops = candidate.anchor.loops;
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
        if (parts.back().siblings) {
          report->emitError("interconnect: ")
              << kind
              << " options switch inside one of the sibling nests that "
                 "replay per pass; not supported";
          return mlir::failure();
        }
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
  if (!parts.empty()) {
    parts.back().end = binding.sequence.size();
  }
  // A part repeats over the loops of its deepest transfer.
  bool mixed = false;
  for (BindingPart &p : parts) {
    for (std::size_t j = p.begin; j < p.end && !p.siblings; ++j) {
      for (const Anchor &a : binding.step_first[j]) {
        if (a.loops.size() > p.loops.size()) {
          p.loops = a.loops;
          p.trips = own_trips(a);
        }
      }
    }
    for (std::size_t j = p.begin; j < p.end; ++j) {
      for (const Anchor &a : binding.step_first[j]) {
        mixed |= p.end - p.begin > 1 && a.loops != p.loops;
      }
    }
  }
  merge_parts(binding, parts, kind);
  if (parts.size() > 1 || mixed) {
    binding.parts = std::move(parts);
  }

  // A second source for a target in one option would silently replace the
  // first in the swb.
  for (std::size_t i = 0; kind == "word" && i < binding.slots.size(); ++i) {
    for (const InterconnectConfigOption &opt : binding.slots[i]) {
      std::map<int, int> source_of;
      for (const InterconnectConfig &c : opt.configs) {
        ConfigKey key = config_key(c);
        auto [it, fresh] = source_of.emplace(key.target, key.source);
        if (!fresh && it->second != key.source) {
          report->emitError("interconnect: swb option ")
              << i << " drives target " << key.target << " from both slot "
              << it->second << " and slot " << key.source;
          return mlir::failure();
        }
      }
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
                                       llvm::StringRef delay, int32_t port,
                                       std::optional<int32_t> dfollows = {}) {
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
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>> params{
      {"delay", builder.getStringAttr(delay)},
      {"iter", builder.getI32IntegerAttr(static_cast<int32_t>(steps.size()))},
      {"step", builder.getI32IntegerAttr(stride)},
      {"port", builder.getI32IntegerAttr(port)}};
  std::string ds = delay.str() + "_ds";
  if (dfollows) {
    params.push_back({"dslope", builder.getStringAttr(ds)});
    params.push_back({"dfollows", builder.getI32IntegerAttr(*dfollows)});
  }
  add_instr(builder, rop.getLoc(), "rep", params);
  return mlir::success();
}

mlir::LogicalResult
emit_sequence_instructions(const InterconnectBinding &binding, RopOp rop,
                           mlir::OpBuilder &builder) {
  builder.setInsertionPoint(rop.getBody().front().getTerminator());

  std::size_t n = binding.sequence.size();
  bool parts = !binding.parts.empty();
  if (n <= 1 && !parts) {
    return mlir::success();
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

  // The parts in the order they run: [begin, end) of the sequence, and the
  // loops it repeats over with every pass.
  struct Span {
    std::size_t begin, end;
    std::vector<uint32_t> trips;
    std::string delay;
  };
  llvm::SmallVector<Span> spans;
  if (!parts) {
    mlir::FailureOr<std::vector<uint32_t>> trips = pass_trips(binding, rop);
    if (mlir::failed(trips)) {
      return mlir::failure();
    }
    spans.push_back({0, n, *trips, delay_name});
  }
  for (std::size_t s = 0; s < binding.parts.size(); ++s) {
    const BindingPart &p = binding.parts[s];
    if (mlir::failed(check_part_trips(binding, p, rop))) {
      return mlir::failure();
    }
    spans.push_back(
        {p.begin, p.end, p.trips, delay_name + "_part" + std::to_string(s)});
  }

  // One evt plus reps per part, then all trans: the AGU reads a rep after a
  // trans as repeating every part.
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

    // AGU Slope Levels: a transfer loop inside the pass loops whose trip
    // count follows pass loop `f` makes every pass from `f` in grow, so the
    // walk's levels from `f` in get a delay slope the scheduler solves
    // (dfollows: levels up to `f`). A sloped pass loop itself, or growth from
    // two different pass loops, is not supported.
    std::size_t T = span.trips.size();
    std::optional<std::size_t> grow_from;
    for (std::size_t j = span.begin; j < span.end; ++j) {
      for (const Anchor &a : binding.step_first[j]) {
        for (std::size_t d = 0; d < a.loop_slope.size(); ++d) {
          if (!loop_sloped(a, d)) {
            continue;
          }
          if (d < T) {
            return rop->emitError("interconnect: a loop the route options "
                                  "repeat over has a sloped trip count; not "
                                  "supported");
          }
          if (a.loop_follows[d] > d) {
            continue;
          }
          std::size_t f = d - a.loop_follows[d];
          if (f >= T) {
            continue; // follows a loop inside the transfer: its own business
          }
          if (grow_from && *grow_from != f) {
            return rop->emitError("interconnect: the passes of the route "
                                  "options grow with two different loops; "
                                  "not supported");
          }
          grow_from = f;
        }
      }
    }
    std::optional<int32_t> stride_dfollows;
    if (grow_from && steps.size() > 1) {
      stride_dfollows = static_cast<int32_t>(T - *grow_from);
    }
    if (mlir::failed(emit_stride(steps, rop, builder, span.delay, port,
                                 stride_dfollows))) {
      return mlir::failure();
    }

    // One level per pass loop, innermost first, replaying the repeated part.
    // Each has a delay of its own: the route fires fewer events per pass than
    // the rops it serves, so it cannot share their period. A part of one
    // option stays configured across every pass and needs none.
    if (steps.size() > 1) {
      for (std::size_t k = span.trips.size(); k-- > 0;) {
        std::string d = span.delay + "_" + std::to_string(k);
        llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>> params{
            {"delay", builder.getStringAttr(d)},
            {"iter", builder.getI32IntegerAttr(
                         static_cast<int32_t>(span.trips[k]))},
            {"step", builder.getI32IntegerAttr(0)},
            {"port", builder.getI32IntegerAttr(port)}};
        std::string ds = d + "_ds";
        if (grow_from && k >= *grow_from) {
          params.push_back({"dslope", builder.getStringAttr(ds)});
          params.push_back({"dfollows", builder.getI32IntegerAttr(
                                            static_cast<int32_t>(k - *grow_from))});
        }
        add_instr(builder, loc, "rep", params);
      }
    }
  }
  for (std::size_t s = 1; s < spans.size(); ++s) {
    add_instr(builder, loc, "trans",
              {{"delay", builder.getStringAttr(delay_name + "_trans" +
                                               std::to_string(s))},
               {"port", builder.getI32IntegerAttr(port)}});
  }
  return mlir::success();
}

// From a transfer to a later step: the data is on the route `a.delay` cycles
// after the anchor's event.
static DelayAttr after(mlir::MLIRContext *ctx, const Anchor &a) {
  return DelayAttr::get(ctx, 1 + a.delay, std::nullopt);
}

// What the debug dumps show against a constraint: the pass that emitted it and
// the rule that fired.
static mlir::ArrayAttr debug_attr(mlir::OpBuilder &builder,
                                  llvm::StringRef rule) {
  return builder.getStrArrayAttr({"InterconnectPass", rule});
}

// Route constraints of a binding in parts (part = MT index on `rop`). Within a
// part as for one part; between parts, the last pass of one is done before the
// next part's evt.
static void emit_part_constraints(const InterconnectBinding &binding,
                                  RopOp rop, mlir::OpBuilder &builder,
                                  const llvm::StringSet<> &config_rops) {
  mlir::Location loc = rop.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  auto rop_ref = mlir::FlatSymbolRefAttr::get(ctx, rop.getSymName());
  auto delay = DelayAttr::get(ctx, 1, std::nullopt);
  std::size_t n = binding.sequence.size();
  llvm::ArrayRef<uint32_t> none;

  auto part_of = [&](std::size_t j) {
    std::size_t p = 0;
    while (j >= binding.parts[p].end) {
      ++p;
    }
    return p;
  };
  // Whether the rop repeats part `p`'s evt over its passes: only a part that
  // switches between options does (emit_sequence_instructions).
  auto has_passes = [&](std::size_t p) {
    const BindingPart &part = binding.parts[p];
    return part.end - part.begin > 1 && !part.trips.empty();
  };
  auto first_pass = [](llvm::ArrayRef<uint32_t> trips) {
    return std::vector<uint32_t>(trips.size(), 0);
  };
  auto last_pass = [](llvm::ArrayRef<uint32_t> trips) {
    std::vector<uint32_t> last;
    for (uint32_t trip : trips) {
      last.push_back(trip - 1);
    }
    return last;
  };

  auto range = [&](mlir::FlatSymbolRefAttr id, llvm::ArrayRef<uint32_t> or_idx,
                   uint32_t mt, llvm::ArrayRef<uint32_t> base,
                   llvm::ArrayRef<uint32_t> from, llvm::ArrayRef<uint32_t> to,
                   llvm::ArrayRef<uint32_t> tail, const Anchor *a = nullptr,
                   llvm::ArrayRef<uint8_t> tail_last = {}) {
    std::vector<uint32_t> lo;
    std::vector<uint32_t> hi;
    for (std::size_t k = 0; k < from.size(); ++k) {
      lo.push_back(base[k] + from[k]);
      hi.push_back(base[k] + to[k]);
    }
    lo.insert(lo.end(), tail.begin(), tail.end());
    hi.insert(hi.end(), tail.begin(), tail.end());
    return sloped_range(ctx, id, or_idx, mt, lo, hi, a, from.size(), tail_last);
  };
  // Step `j` of the rop over passes `from`..`to` of its part (ignored for an
  // evt that does not repeat over them).
  auto step_at = [&](std::size_t j, llvm::ArrayRef<uint32_t> from,
                     llvm::ArrayRef<uint32_t> to) {
    std::size_t p = part_of(j);
    const BindingPart &part = binding.parts[p];
    std::vector<uint32_t> tail;
    if (part.end - part.begin > 1) {
      tail.push_back(static_cast<uint32_t>(j - part.begin));
    }
    uint32_t mt = static_cast<uint32_t>(p);
    if (!has_passes(p)) {
      return range(rop_ref, {}, mt, none, none, none, tail);
    }
    return range(rop_ref, {}, mt, first_pass(part.trips), from, to, tail);
  };
  // A transfer over passes `from`..`to` of its loops; a transfer that runs
  // once has none. Its loops inside those at their first pass, or with
  // `last` at their last.
  auto use_at = [&](const Anchor &a, llvm::ArrayRef<uint32_t> from,
                    llvm::ArrayRef<uint32_t> to, bool last = false) {
    llvm::ArrayRef<uint32_t> ir(a.ir_idx);
    if (own_trips(a).empty()) {
      std::vector<uint8_t> tail_last(ir.size(), a.is_last);
      return range(a.instr_id, a.or_idx, a.mt, none, none, none, ir, &a,
                   tail_last);
    }
    std::vector<uint32_t> tail(ir.begin() + std::min(from.size(), ir.size()),
                               ir.end());
    std::vector<uint8_t> tail_last(tail.size(), a.is_last);
    for (std::size_t k = 0; last && k < tail.size(); ++k) {
      if (from.size() + k < a.pass_hi.size()) {
        tail[k] = a.pass_hi[from.size() + k];
        tail_last[k] = true;
      }
    }
    return range(a.instr_id, a.or_idx, a.mt, ir, from, to, tail, &a, tail_last);
  };
  // A transfer's first or last pass through its own loops.
  auto use_first = [&](const Anchor &a) {
    std::vector<uint32_t> first = first_pass(own_trips(a));
    return use_at(a, first, first);
  };
  auto use_last = [&](const Anchor &a) {
    std::vector<uint32_t> last = last_pass(own_trips(a));
    return use_at(a, last, last);
  };
  // Transfer `a` over passes `from`..`to` of part `p`, and the passes of the
  // part to pair it with. One over fewer loops than the part runs before or
  // after the deeper ones, so only in their first or last pass; nothing when
  // that pass is outside `from`..`to`.
  struct Paired {
    AnchorRangeAttr use;
    std::vector<uint32_t> from, to;
  };
  auto pair_use = [&](const Anchor &a, std::size_t p,
                      llvm::ArrayRef<uint32_t> from,
                      llvm::ArrayRef<uint32_t> to,
                      bool last = false) -> std::optional<Paired> {
    const BindingPart &part = binding.parts[p];
    std::size_t d = std::min(a.loops.size(), from.size());
    bool after = false;
    for (std::size_t j = part.begin; j < part.end && d < from.size(); ++j) {
      for (const Anchor &x : binding.step_first[j]) {
        after |= x.loops.size() > d && x.pos < a.pos;
      }
    }
    Paired out{use_at(a, from.take_front(d), to.take_front(d), last),
               std::vector<uint32_t>(from.begin(), from.end()),
               std::vector<uint32_t>(to.begin(), to.end())};
    for (std::size_t k = d; k < from.size(); ++k) {
      uint32_t at = after ? part.trips[k] - 1 : 0;
      if (at < from[k] || at > to[k]) {
        return std::nullopt;
      }
      out.from[k] = out.to[k] = at;
    }
    return out;
  };
  auto add = [&](AnchorRangeAttr src, AnchorRangeAttr dst, llvm::StringRef rule,
                 DelayAttr d = {}) {
    CstrOp::create(builder, loc, src, dst, d ? d : delay,
                   builder.getBoolAttr(false))
        ->setAttr("debug", debug_attr(builder, rule));
  };

  // config -> first_use: the step that switches an option in comes before the
  // transfers that use it -- on every pass when the step repeats, and before
  // the first pass when its option stays on through all of them.
  for (std::size_t j = 0; j < n; ++j) {
    std::size_t p = part_of(j);
    const BindingPart &part = binding.parts[p];
    for (const Anchor &a : binding.step_first[j]) {
      if (a.instr_id && config_rops.contains(a.instr_id.getValue())) {
        continue;
      }
      if (has_passes(p)) {
        std::vector<uint32_t> first = first_pass(part.trips);
        std::vector<uint32_t> last = last_pass(part.trips);
        if (auto u = pair_use(a, p, first, last)) {
          add(step_at(j, u->from, u->to), u->use, "route-first-use");
        }
      } else {
        add(step_at(j, none, none), use_first(a), "route-first-use");
      }
    }
  }

  // last_use -> next config: per pass within a part, last pass across parts.
  for (std::size_t j = 1; j < n; ++j) {
    std::size_t p = part_of(j);
    bool same_part = part_of(j - 1) == p;
    if (same_part && binding.sequence[j - 1] == binding.sequence[j]) {
      continue;
    }
    for (const Anchor &a : binding.step_last[j - 1]) {
      // A transfer carried into the next option spans the switch; see the
      // rule of the same name in emit_interconnect_constraints.
      if (llvm::is_contained(binding.step_first[j], a)) {
        continue;
      }
      if (same_part && has_passes(p)) {
        const BindingPart &part = binding.parts[p];
        std::vector<uint32_t> first = first_pass(part.trips);
        std::vector<uint32_t> last = last_pass(part.trips);
        if (auto u = pair_use(a, p, first, last, /*last=*/true)) {
          add(u->use, step_at(j, u->from, u->to), "route-last-use",
              after(ctx, a));
        }
      } else if (same_part) {
        add(use_at(a, none, none), step_at(j, none, none), "route-last-use",
            after(ctx, a));
      } else {
        std::vector<uint32_t> first = first_pass(binding.parts[p].trips);
        add(use_last(a), step_at(j, first, first), "route-last-use",
            after(ctx, a));
      }
    }
  }

  // A part that repeats starts over on the next pass, so its last option's
  // last use is done before that pass's first step; one constraint per pass
  // loop, as in emit_interconnect_constraints.
  for (std::size_t p = 0; p < binding.parts.size(); ++p) {
    if (!has_passes(p)) {
      continue;
    }
    const BindingPart &part = binding.parts[p];
    std::vector<uint32_t> first = first_pass(part.trips);
    std::vector<uint32_t> last = last_pass(part.trips);
    for (std::size_t k = 0; k < part.trips.size(); ++k) {
      if (part.trips[k] < 2) {
        continue;
      }
      std::vector<uint32_t> src_from = first;
      std::vector<uint32_t> src_to = last;
      src_to[k] = last[k] - 1;
      for (std::size_t inner = k + 1; inner < part.trips.size(); ++inner) {
        src_from[inner] = last[inner];
      }
      for (const Anchor &a : binding.step_last[part.end - 1]) {
        auto u = pair_use(a, p, src_from, src_to, /*last=*/true);
        if (!u) {
          continue;
        }
        // The pass after: one on in loop k, loops inside it at their first.
        std::vector<uint32_t> dst_from = u->from;
        std::vector<uint32_t> dst_to = u->to;
        ++dst_from[k];
        ++dst_to[k];
        for (std::size_t inner = k + 1; inner < part.trips.size(); ++inner) {
          dst_from[inner] = dst_to[inner] = 0;
        }
        add(u->use, step_at(part.begin, dst_from, dst_to), "route-next-pass",
            after(ctx, a));
      }
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
  if (has_sequence && binding.parts.empty()) {
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
                      llvm::ArrayRef<uint32_t> tail, const Anchor *a = nullptr) {
    std::vector<uint32_t> lo;
    std::vector<uint32_t> hi;
    for (std::size_t k = 0; k < from.size(); ++k) {
      lo.push_back(base[k] + from[k]);
      hi.push_back(base[k] + to[k]);
    }
    lo.insert(lo.end(), tail.begin(), tail.end());
    hi.insert(hi.end(), tail.begin(), tail.end());
    std::vector<uint8_t> tail_last(tail.size(), a && a->is_last);
    return sloped_range(ctx, id, or_idx, mt, lo, hi, a, from.size(), tail_last);
  };
  // A transfer's anchor, and a step of this rop's sequence, over those passes.
  auto of_use = [&](const Anchor &a, llvm::ArrayRef<uint32_t> from,
                    llvm::ArrayRef<uint32_t> to) {
    llvm::ArrayRef<uint32_t> ir(a.ir_idx);
    return spanning(a.instr_id, a.or_idx, a.mt, ir, from, to,
                    ir.drop_front(from.size()), &a);
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

  if (!binding.parts.empty()) {
    emit_part_constraints(binding, rop, builder, config_rops);
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
        CstrOp::create(builder, loc, src_ar, dst_ar, after(ctx, a),
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
                       of_step(first_step, dst_from, dst_to), after(ctx, a),
                       builder.getBoolAttr(false))
            ->setAttr("debug", debug_attr(builder, "route-next-pass"));
      }
    }
  }
}

} // namespace vesyla::pasm
