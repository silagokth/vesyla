#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include "vesyla/Dialect/Pasm/Transforms/NoReconfigurationPass.hpp"
#include "vesyla/Support/Config.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace vesyla::pasm {
#define GEN_PASS_DEF_NORECONFIGURATIONPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Ports per slot, the way `act` and the component map number them: port index
// `slot * 4 + port`. It is the sequencer's FSM_PER_SLOT.
constexpr int ports_per_slot = 4;

// A cell (row, col) and a slot in it.
using SlotKey = std::tuple<int, int, int>;

// One AGU: a port of a slot of cell (row, col).
using AguKey = std::tuple<int, int, int, int>;

// What the architecture says about one resource.
struct ResourceEntry {
  int row;
  int col;
  int slot;
  std::string kind;
  // its parameters in the architecture
  nlohmann::json parameters;
  // cycles from an act to the first address of its AGUs
  int agu_latency;
};

// Bit widths of the rep fields of one kind of resource, from its ISA. A base
// rep (ext = 0) carries the lower half of each field and its repx (ext = 1)
// the upper half, so the upper half starts at this width (agu_controller's
// DELAY_HI, ITER_HI and STEP_HI).
struct RepWidths {
  int iter = 0;
  int step = 0;
  int delay = 0;
};

// One level of a repetition: a rep's iter, step and delay at full width.
struct Level {
  uint64_t iter = 0;
  uint64_t step = 0;
  uint64_t delay = 0;
};

// Write one rep field: the lower half from a base rep, the upper half from its
// repx, leaving the other half as it was.
void write_half(uint64_t &field, uint64_t value, int width, bool ext) {
  uint64_t mask = (1ULL << width) - 1;
  if (ext) {
    field = (field & mask) | ((value & mask) << width);
  } else {
    field = (field & ~mask) | (value & mask);
  }
}

// One configuration of one AGU, the values agu_controller holds when the AGU is
// started. It replays the decoding SizeAguPass sizes for: an `evt` opens the
// next lane, whose IR level 0 defaults to one iteration, and restarts the level
// count. A base `rep` writes the next level: an IR level of the current lane
// until a `trans` has been seen, an OR level after it; its `repx` writes the
// upper half of the level the base wrote. A `trans` writes the next MT entry
// and sends every later `rep` to OR; it is dropped once an OR level was
// written. As in SizeAguPass, the IR levels are taken to be sized to the
// deepest lane, so none spills into OR.
class AguConfig {
public:
  explicit AguConfig(RepWidths widths) : widths_(widths) {}

  void evt(uint64_t init_addr) {
    lanes_.push_back(Lane{init_addr, {Level{1, 0, 0}}});
    level_ = 0;
  }

  // False when no `evt` has opened a lane of this configuration.
  bool rep(bool ext, uint64_t iter, uint64_t step, uint64_t delay) {
    if (lanes_.empty()) {
      return false;
    }
    std::vector<Level> &levels = use_or_ ? or_ : lanes_.back().ir;
    size_t level = ext ? (level_ == 0 ? 0 : level_ - 1) : level_;
    if (levels.size() <= level) {
      levels.resize(level + 1);
    }
    write_half(levels[level].iter, iter, widths_.iter, ext);
    write_half(levels[level].step, step, widths_.step, ext);
    write_half(levels[level].delay, delay, widths_.delay, ext);
    if (use_or_) {
      or_configured_ = true;
    }
    if (!ext) {
      level_++;
    }
    return true;
  }

  // False when no `evt` has opened a lane of this configuration.
  bool trans(uint64_t delay) {
    if (lanes_.empty()) {
      return false;
    }
    if (or_configured_) {
      return true;
    }
    mt_.push_back(delay);
    use_or_ = true;
    level_ = 0;
    return true;
  }

  // Every address a run of this configuration generates: per lane, its initial
  // address plus each IR level's iteration count times its step, and around
  // all lanes each OR level's. An AGU no `evt` configured generates address 0
  // alone. Nothing when an address exceeds `limit`.
  std::optional<std::set<uint64_t>> addresses(uint64_t limit) const {
    auto expand = [&](std::set<uint64_t> &addrs, const Level &level) {
      std::set<uint64_t> next;
      for (uint64_t addr : addrs) {
        for (uint64_t k = 0; k < std::max<uint64_t>(level.iter, 1); k++) {
          uint64_t expanded = addr + k * level.step;
          if (expanded > limit) {
            return false;
          }
          next.insert(expanded);
          if (level.step == 0) {
            break;
          }
        }
      }
      addrs = std::move(next);
      return true;
    };

    std::set<uint64_t> all;
    if (lanes_.empty()) {
      all.insert(0);
    }
    for (const Lane &lane : lanes_) {
      std::set<uint64_t> addrs = {lane.init_addr};
      for (const Level &level : lane.ir) {
        if (!expand(addrs, level)) {
          return std::nullopt;
        }
      }
      all.insert(addrs.begin(), addrs.end());
    }
    for (const Level &level : or_) {
      if (!expand(all, level)) {
        return std::nullopt;
      }
    }
    for (uint64_t addr : all) {
      if (addr > limit) {
        return std::nullopt;
      }
    }
    return all;
  }

  // The address a run ends on, which the resource keeps selecting after it:
  // the last iteration of every level of the last lane and of every OR level.
  uint64_t last_address() const {
    if (lanes_.empty()) {
      return 0;
    }
    uint64_t addr = lanes_.back().init_addr;
    for (const Level &level : lanes_.back().ir) {
      addr += (std::max<uint64_t>(level.iter, 1) - 1) * level.step;
    }
    for (const Level &level : or_) {
      addr += (std::max<uint64_t>(level.iter, 1) - 1) * level.step;
    }
    return addr;
  }

  nlohmann::json to_json() const {
    auto level_json = [](const Level &level) {
      return nlohmann::json{{"iter", level.iter},
                            {"step", level.step},
                            {"delay", level.delay}};
    };
    nlohmann::json lanes = nlohmann::json::array();
    for (const Lane &lane : lanes_) {
      nlohmann::json ir = nlohmann::json::array();
      for (const Level &level : lane.ir) {
        ir.push_back(level_json(level));
      }
      lanes.push_back({{"init_addr", lane.init_addr}, {"ir", ir}});
    }
    nlohmann::json mt = nlohmann::json::array();
    for (uint64_t delay : mt_) {
      mt.push_back({{"delay", delay}});
    }
    nlohmann::json or_levels = nlohmann::json::array();
    for (const Level &level : or_) {
      or_levels.push_back(level_json(level));
    }
    return {{"lanes", lanes}, {"mt", mt}, {"or", or_levels}};
  }

private:
  struct Lane {
    uint64_t init_addr;
    std::vector<Level> ir;
  };

  RepWidths widths_;
  std::vector<Lane> lanes_;
  std::vector<uint64_t> mt_;
  std::vector<Level> or_;
  size_t level_ = 0;
  bool use_or_ = false;
  bool or_configured_ = false;
};

// How a kind of resource addresses its conf entries: a conf writes the entry
// named by the `key` fields with the `value` fields, as the resource's RTL
// decodes it. `variant` is the conf variant, empty for a conf without any.
//
// The entries of an option table are grouped into options by the conf's
// `option` field, and the resource uses the option the AGU of port
// `selector_port` addresses; `table` names the table. A conf of a kind without
// options (`selector_port` -1) writes its entry directly. The resource
// switches to an option `select_latency` cycles after the selector presents
// its address.
struct ConfRule {
  const char *kind;
  const char *variant;
  const char *table;
  int selector_port;
  int select_latency;
  std::vector<std::string> key;
  std::vector<std::string> value;
};

const std::vector<ConfRule> &conf_rules() {
  static const std::vector<ConfRule> rules = {
      // config_regs[option] = {mode, immediate}, selected by AGU 0 in the
      // cycle it presents the address
      {"dpu", "", "mode", 0, 0, {}, {"mode", "immediate"}},
      // slot_link[target] of crossbar option `option` = source, selected by
      // AGU_SWB_INDEX through the registered current_option
      {"swb", "swb", "crossbar", 0, 1, {"target"}, {"source"}},
      // send_links / receive_links[source] of route option `option` = target,
      // selected by AGU_ROUTE_INDEX through the registered current_option
      {"swb", "route", "route", 1, 1, {"sr", "source"}, {"target"}},
      // memory[address] = value
      {"rf", "", "memory", -1, 0, {"address"}, {"value"}},
      {"rf_row", "", "memory", -1, 0, {"address"}, {"value"}},
  };
  return rules;
}

// One version of one option of an option table: the entries the option holds
// from the conf that starts it until a conf changes it. Until an activation
// reads the version, a conf adding an entry, or rewriting one the version still
// has from the one before it, belongs to the version: the confs of one
// reconfiguration write several entries one after the other. Once an
// activation has read it, any change starts a new version.
struct OptionVersion {
  // entry key -> the fields of the conf that wrote it
  std::map<std::string, nlohmann::json> entries;
  // the keys a conf wrote while this version was current
  std::set<std::string> written;
  // the conf that started this version, and the cycle it issued in
  InstrOp first;
  uint64_t cycle = 0;
  // an activation of the selector reads this version
  bool activated = false;
  // the resource selects this option while this version is current
  bool held = false;
};

// One activation of the AGU that selects an option table, and what it reads.
struct OptionUse {
  // the configuration of the selector this activation runs, by its index in
  // the selector's list of configurations
  size_t config;
  // address -> (option, version) it reads; no version for an option never
  // written
  std::map<uint64_t, std::pair<int, std::optional<size_t>>> reads;
  InstrOp act;
};

// One option table of one resource, replayed over the program.
struct OptionTable {
  const ConfRule *rule = nullptr;
  int num_options = 0;
  std::map<int, std::vector<OptionVersion>> options;
  // the option the resource selects: 0 from reset, and after a run the
  // address the run ended on
  int held = 0;
  std::vector<OptionUse> uses;
  // after allocation: the entries each option of the table holds, for the
  // whole program
  std::vector<std::optional<std::map<std::string, nlohmann::json>>> slots;
};

// The conf state of one resource: the entries of kinds without options, by
// key, and the option tables, by name.
struct ResourceConf {
  std::map<std::string, std::pair<InstrOp, nlohmann::json>> fixed;
  std::map<std::string, OptionTable> tables;
};

std::optional<int64_t> int_param(InstrOp instr, llvm::StringRef name) {
  auto attr = instr.getParam().getAs<mlir::IntegerAttr>(name);
  if (!attr) {
    return std::nullopt;
  }
  return attr.getInt();
}

std::string string_param(InstrOp instr, llvm::StringRef name) {
  auto attr = instr.getParam().getAs<mlir::StringAttr>(name);
  return attr ? attr.getValue().str() : "";
}

// The fields of an instruction as JSON, without the slot and port that only
// say where it is sent.
nlohmann::json params_json(InstrOp instr) {
  nlohmann::json fields = nlohmann::json::object();
  for (mlir::NamedAttribute attr : instr.getParam()) {
    llvm::StringRef name = attr.getName().getValue();
    if (name == "slot" || name == "port") {
      continue;
    }
    if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr.getValue())) {
      fields[name.str()] = int_attr.getInt();
    } else if (auto str_attr =
                   llvm::dyn_cast<mlir::StringAttr>(attr.getValue())) {
      fields[name.str()] = str_attr.getValue().str();
    }
  }
  return fields;
}

// The sequencer's scalar registers, replayed to know the port map a mode-2
// `act` reads. A register written by a calculation this does not replay is
// unknown.
class Registers {
public:
  Registers(int count, int width)
      : regs_(count, 0), width_(width),
        mask_(width >= 64 ? ~0ULL : (1ULL << width) - 1) {}

  // False when the calculation names a register the sequencer does not have.
  bool calc(InstrOp instr) {
    auto mode = int_param(instr, "mode");
    auto operand1 = int_param(instr, "operand1");
    auto operand2_sd = int_param(instr, "operand2_sd");
    auto operand2 = int_param(instr, "operand2");
    auto result = int_param(instr, "result");
    if (!mode || !operand1 || !operand2_sd || !operand2 || !result ||
        !valid(*operand1) || !valid(*result) ||
        (*operand2_sd != 0 && !valid(*operand2))) {
      return false;
    }

    std::optional<uint64_t> in1 = regs_[*operand1];
    std::optional<uint64_t> in2 = *operand2_sd == 0
                                      ? std::optional<uint64_t>(*operand2)
                                      : regs_[*operand2];
    std::optional<uint64_t> out;
    if (in1 && in2) {
      uint64_t half = (1ULL << (width_ / 2)) - 1;
      if (*mode == 1) { // add
        out = *in1 + *in2;
      } else if (*mode == 2) { // sub
        out = *in1 - *in2;
      } else if (*mode == 23) { // addh: add to the upper half
        uint64_t upper = ((*in1 >> (width_ / 2)) + *in2) & half;
        out = (upper << (width_ / 2)) | (*in1 & half);
      }
    }
    regs_[*result] = out ? std::optional<uint64_t>(*out & mask_) : out;
    return true;
  }

  // The 64-bit port map {r[first+3], r[first+2], r[first+1], r[first]}.
  std::optional<uint64_t> port_map(int64_t first) const {
    uint64_t map = 0;
    for (int i = 0; i * width_ < 64; i++) {
      if (!valid(first + i) || !regs_[first + i]) {
        return std::nullopt;
      }
      map |= *regs_[first + i] << (width_ * i);
    }
    return map;
  }

private:
  bool valid(int64_t index) const {
    return index >= 0 && index < static_cast<int64_t>(regs_.size());
  }

  std::vector<std::optional<uint64_t>> regs_;
  int width_;
  uint64_t mask_;
};

// The port mask, per slot, an `act` raises, decoded the way the sequencer does:
// a 4-bit mask per slot, slots past the 64 activate wires dropped.
std::optional<std::map<int, int>> activated_masks(InstrOp act,
                                                  const Registers &regs) {
  auto mode = int_param(act, "mode");
  auto param = int_param(act, "param");
  auto ports = int_param(act, "ports");
  if (!mode || !param || !ports) {
    return std::nullopt;
  }

  uint64_t flat = 0;
  uint64_t mask = static_cast<uint64_t>(*ports);
  if (*mode == 0) {
    // `ports` shifted to the first port of slot `param`.
    int shift = *param * ports_per_slot;
    flat = shift >= 64 ? 0 : mask << shift;
  } else if (*mode == 1) {
    // `param` to every slot set in `ports`.
    for (int slot = 0; slot * ports_per_slot < 64; slot++) {
      if (mask & (1ULL << slot)) {
        flat |= static_cast<uint64_t>(*param & 0xF) << (slot * ports_per_slot);
      }
    }
  } else if (*mode == 2) {
    // The port map held by the registers from `param` on.
    auto map = regs.port_map(*param);
    if (!map) {
      return std::nullopt;
    }
    flat = *map;
  } else {
    return std::nullopt;
  }

  std::map<int, int> masks;
  for (int slot = 0; slot * ports_per_slot < 64; slot++) {
    int slot_mask = (flat >> (slot * ports_per_slot)) & 0xF;
    if (slot_mask != 0) {
      masks[slot] = slot_mask;
    }
  }
  return masks;
}

class NoReconfigurationPass
    : public impl::NoReconfigurationPassBase<NoReconfigurationPass> {
  // Per cell: the ports each cycle's activation raises, per slot, and the
  // cycle of the halt.
  std::map<std::pair<int, int>, std::map<uint64_t, std::map<int, int>>>
      activations_;
  std::map<std::pair<int, int>, uint64_t> lengths_;

public:
  using impl::NoReconfigurationPassBase<
      NoReconfigurationPass>::NoReconfigurationPassBase;

  void runOnOperation() final {
    Config cfg;
    nlohmann::json arch = cfg.get_arch_json();
    nlohmann::json isa = cfg.get_isa_json();

    // The rep field widths of every kind of resource the ISA has.
    std::map<std::string, RepWidths> rep_widths;
    for (const auto &component : isa["components"]) {
      if (!component.contains("kind")) {
        continue;
      }
      for (const auto &instruction : component["instructions"]) {
        if (instruction["name"] != "rep") {
          continue;
        }
        RepWidths widths;
        for (const auto &segment : instruction["segments"]) {
          if (segment["name"] == "iter") {
            widths.iter = segment["bitwidth"];
          } else if (segment["name"] == "step") {
            widths.step = segment["bitwidth"];
          } else if (segment["name"] == "delay") {
            widths.delay = segment["bitwidth"];
          }
        }
        rep_widths[component["kind"]] = widths;
      }
    }

    // How many options each kind of option table has: as many as the conf's
    // `option` field can name.
    std::map<std::pair<std::string, std::string>, int> option_counts;
    for (const auto &component : isa["components"]) {
      if (!component.contains("kind")) {
        continue;
      }
      for (const auto &instruction : component["instructions"]) {
        if (instruction["name"] != "conf") {
          continue;
        }
        auto add = [&](const std::string &variant,
                       const nlohmann::json &segments) {
          for (const auto &segment : segments) {
            if (segment["name"] == "option") {
              option_counts[{component["kind"], variant}] =
                  1 << segment["bitwidth"].get<int>();
            }
          }
        };
        add("", instruction.value("segments", nlohmann::json::array()));
        for (const auto &variant :
             instruction.value("variants", nlohmann::json::array())) {
          add(variant["name"],
              variant.value("segments", nlohmann::json::array()));
        }
      }
    }

    // Every cell and resource in the architecture, and which resource
    // occupies each slot. Anything addressed to a slot no resource occupies is
    // dispatched nowhere, and so has nothing to hardcode.
    std::map<std::pair<int, int>, nlohmann::json> controllers;
    std::vector<ResourceEntry> resources;
    std::map<SlotKey, size_t> resource_at;
    for (const auto &cell : arch["cells"]) {
      int row = cell["coordinates"]["row"];
      int col = cell["coordinates"]["col"];
      controllers[{row, col}] = cell["cell"]["controller"];
      for (const auto &resource : cell["cell"]["resources_list"]) {
        int first_slot = resource["slot"];
        int size = resource["size"];
        for (int slot = first_slot; slot < first_slot + size; slot++) {
          resource_at[{row, col, slot}] = resources.size();
        }
        nlohmann::json parameters =
            resource.value("parameters", nlohmann::json::object());
        resources.push_back({row, col, first_slot,
                             resource.value("kind", std::string()), parameters,
                             parameters.value("AGU_LATENCY", 1)});
      }
    }

    // Per resource, every configuration each of its AGUs runs, in the order it
    // is activated, and its conf entries by key.
    std::vector<std::map<std::pair<int, int>, nlohmann::json>> agu_configs(
        resources.size());
    std::vector<ResourceConf> confs(resources.size());

    std::map<std::pair<int, int>, RawOp> streams;
    bool failed = false;
    getOperation().walk([&](RawOp raw) {
      auto [it, inserted] = streams.insert({{raw.getRow(), raw.getCol()}, raw});
      if (!inserted) {
        raw.emitError("no-reconfiguration: cell (")
            << raw.getRow() << ", " << raw.getCol()
            << ") has more than one instruction stream";
        failed = true;
      }
    });
    if (failed) {
      signalPassFailure();
      return;
    }

    for (const auto &[coord, controller] : controllers) {
      auto [row, col] = coord;
      auto stream = streams.find(coord);
      bool empty = true;
      if (stream != streams.end()) {
        for (InstrOp instr :
             stream->second.getBody().front().getOps<InstrOp>()) {
          if (instr.getType() != "halt") {
            empty = false;
            break;
          }
        }
      }
      if (empty) {
        getOperation().emitError("no-reconfiguration: cell (")
            << row << ", " << col
            << ") has nothing to run; a fabric without reconfiguration "
               "cannot have an unused cell yet";
        failed = true;
        continue;
      }
      if (mlir::failed(replay_cell(stream->second, controller, rep_widths,
                                   option_counts, resources, resource_at,
                                   agu_configs, confs))) {
        failed = true;
      }
    }
    if (failed) {
      signalPassFailure();
      return;
    }

    for (size_t i = 0; i < resources.size(); i++) {
      for (auto &[name, table] : confs[i].tables) {
        if (mlir::failed(allocate_options(
                table, resources[i], agu_configs[i],
                activations_[{resources[i].row, resources[i].col}]))) {
          failed = true;
        }
      }
    }
    if (failed) {
      signalPassFailure();
      return;
    }

    for (size_t i = 0; i < resources.size(); i++) {
      const ResourceEntry &resource = resources[i];
      nlohmann::json agus = nlohmann::json::array();
      for (const auto &[slot_port, configs] : agu_configs[i]) {
        agus.push_back({{"slot", slot_port.first},
                        {"port", slot_port.second},
                        {"configs", configs}});
      }
      nlohmann::json conf = nlohmann::json::array();
      for (const auto &[key, entry] : confs[i].fixed) {
        conf.push_back(entry.second);
      }
      // Each option the allocation filled, with the entries of the version
      // it holds, renumbered to that option.
      for (const auto &[name, table] : confs[i].tables) {
        for (size_t option = 0; option < table.slots.size(); option++) {
          if (!table.slots[option]) {
            continue;
          }
          for (const auto &[key, fields] : *table.slots[option]) {
            nlohmann::json entry = fields;
            entry["option"] = option;
            conf.push_back(entry);
          }
        }
      }

      // The AGUs have to be built for the configurations they run, including
      // those added to switch options, so the sizes SizeAguPass set only grow,
      // and only where the resource has the parameter.
      uint64_t number_ir = 1, number_mt = 0, number_or = 0, agu_used = 0;
      for (const auto &[slot_port, configs] : agu_configs[i]) {
        agu_used |= 1ULL << slot_port.second;
        for (const auto &config : configs) {
          uint64_t lanes = config["lanes"].size();
          number_mt = std::max<uint64_t>(
              number_mt, std::max<uint64_t>(lanes == 0 ? 0 : lanes - 1,
                                            config["mt"].size()));
          number_or = std::max<uint64_t>(number_or, config["or"].size());
          for (const auto &lane : config["lanes"]) {
            number_ir = std::max<uint64_t>(number_ir, lane["ir"].size());
          }
        }
      }
      nlohmann::json parameters = resource.parameters;
      auto grow = [&](const char *name, uint64_t need, bool merge_bits) {
        if (!parameters.contains(name)) {
          return;
        }
        uint64_t had = parameters[name].get<uint64_t>();
        parameters[name] = merge_bits ? (had | need) : std::max(had, need);
      };
      grow("NUMBER_IR", number_ir, false);
      grow("NUMBER_MT", number_mt, false);
      grow("NUMBER_OR", number_or, false);
      grow("AGU_USED", agu_used, true);

      cfg.set_noreconf_resource_fields(
          resource.row, resource.col, resource.slot,
          {{"parameters", parameters},
           {"static_config", {{"agus", agus}, {"conf", conf}}}});
    }

    // Each controller issues the activations of its cell, the program's and
    // those added to switch options, and halts at the cycle its program does.
    for (const auto &[coord, length] : lengths_) {
      nlohmann::json activations = nlohmann::json::array();
      for (const auto &[cycle, masks] : activations_[coord]) {
        nlohmann::json activate = nlohmann::json::array();
        for (auto [slot, mask] : masks) {
          activate.push_back({{"slot", slot}, {"mask", mask}});
        }
        activations.push_back({{"cycle", cycle}, {"activate", activate}});
      }
      cfg.set_noreconf_controller_fields(
          coord.first, coord.second,
          {{"kind", "fsm"}, {"activations", activations}, {"length", length}});
    }
  }

private:
  // Replay the instruction stream of one cell, counting the cycle each
  // instruction issues in: one each, and a wait(N) N + 1. Record its
  // activations and halt cycle on the controller, and its AGU configurations
  // and conf entries on its resources.
  mlir::LogicalResult replay_cell(
      RawOp raw, const nlohmann::json &controller,
      const std::map<std::string, RepWidths> &rep_widths,
      const std::map<std::pair<std::string, std::string>, int> &option_counts,
      const std::vector<ResourceEntry> &resources,
      const std::map<SlotKey, size_t> &resource_at,
      std::vector<std::map<std::pair<int, int>, nlohmann::json>> &agu_configs,
      std::vector<ResourceConf> &confs) {
    int row = raw.getRow();
    int col = raw.getCol();
    nlohmann::json parameters = controller.value("parameters", nlohmann::json{});
    Registers regs(parameters.value("NUM_SCALAR_REGS", 16),
                   parameters.value("SCALAR_REG_WIDTH", 16));

    // The configuration each AGU has been given since it last started, and per
    // resource the AGU its last `evt` addressed: a `rep` or `trans` configures
    // that one, whichever port it names itself.
    std::map<AguKey, std::pair<InstrOp, AguConfig>> pending;
    std::map<size_t, AguKey> current;

    std::map<uint64_t, std::map<int, int>> &activations =
        activations_[{row, col}];
    uint64_t cycle = 0;
    std::optional<uint64_t> length;

    for (InstrOp instr : raw.getBody().front().getOps<InstrOp>()) {
      llvm::StringRef type = instr.getType();
      if (length) {
        return instr.emitError("no-reconfiguration: instruction after the "
                               "halt");
      }

      if (type == "halt") {
        length = cycle;
        continue;
      }
      if (type == "wait") {
        if (int_param(instr, "mode").value_or(0) != 0) {
          return instr.emitError("no-reconfiguration: a wait for events has "
                                 "no static cycle count");
        }
        cycle += int_param(instr, "cycle").value_or(0) + 1;
        continue;
      }
      if (type == "calc") {
        if (!regs.calc(instr)) {
          return instr.emitError("no-reconfiguration: cannot replay this "
                                 "calculation");
        }
        cycle++;
        continue;
      }
      if (type == "act") {
        auto masks = activated_masks(instr, regs);
        if (!masks) {
          return instr.emitError("no-reconfiguration: cannot tell which "
                                 "ports this act starts");
        }
        for (auto [slot, mask] : *masks) {
          activations[cycle][slot] |= mask;
          // An AGU runs the configuration it was given when it is started,
          // and comes back from the run cleared. One that was given none runs
          // the cleared one.
          auto resource = resource_at.find({row, col, slot});
          if (resource == resource_at.end()) {
            continue;
          }
          for (int port = 0; port < ports_per_slot; port++) {
            if (!(mask & (1 << port))) {
              continue;
            }
            auto it = pending.find({row, col, slot, port});
            AguConfig config = it != pending.end() ? it->second.second
                                                   : AguConfig(RepWidths{});
            if (it != pending.end()) {
              pending.erase(it);
            }
            nlohmann::json &configs =
                agu_configs[resource->second][{slot, port}];
            configs.push_back(config.to_json());

            // An AGU that selects an option table reads the options it
            // addresses, and leaves the resource selecting the last one.
            const ResourceEntry &entry = resources[resource->second];
            for (auto &[name, table] : confs[resource->second].tables) {
              if (slot != entry.slot || port != table.rule->selector_port) {
                continue;
              }
              auto addrs = config.addresses(table.num_options - 1);
              if (!addrs) {
                return instr.emitError("no-reconfiguration: this act starts "
                                       "an AGU of the ")
                       << entry.kind << " in slot " << entry.slot
                       << " that addresses past its " << table.num_options
                       << " " << name << " options";
              }
              OptionUse use{configs.size() - 1, {}, instr};
              for (uint64_t addr : *addrs) {
                auto versions = table.options.find(addr);
                std::optional<size_t> version;
                if (versions != table.options.end()) {
                  version = versions->second.size() - 1;
                  versions->second.back().activated = true;
                }
                use.reads[addr] = {static_cast<int>(addr), version};
              }
              table.uses.push_back(use);
              table.held = config.last_address();
              auto held = table.options.find(table.held);
              if (held != table.options.end()) {
                held->second.back().held = true;
              }
            }
          }
        }
        cycle++;
        continue;
      }
      if (type == "brn") {
        return instr.emitError("no-reconfiguration: a branch has no static "
                               "cycle count");
      }
      if (type != "evt" && type != "rep" && type != "trans" &&
          type != "conf") {
        return instr.emitError("no-reconfiguration: cannot hardcode a ")
               << type << " instruction";
      }

      cycle++;
      auto slot = int_param(instr, "slot");
      if (!slot) {
        return instr.emitError("no-reconfiguration: instruction has no slot");
      }
      auto resource = resource_at.find({row, col, static_cast<int>(*slot)});
      if (resource == resource_at.end()) {
        continue;
      }
      const ResourceEntry &entry = resources[resource->second];

      if (type == "conf") {
        if (mlir::failed(record_conf(instr, cycle - 1, entry, option_counts,
                                     confs[resource->second]))) {
          return mlir::failure();
        }
        continue;
      }

      auto widths = rep_widths.find(entry.kind);
      if (type == "evt") {
        auto port = int_param(instr, "port");
        if (!port) {
          return instr.emitError("no-reconfiguration: evt has no port");
        }
        AguKey key{row, col, static_cast<int>(*slot), static_cast<int>(*port)};
        current[resource->second] = key;
        auto it = pending.find(key);
        if (it == pending.end()) {
          it = pending
                   .insert({key, {instr, AguConfig(widths != rep_widths.end()
                                                       ? widths->second
                                                       : RepWidths{})}})
                   .first;
        }
        it->second.second.evt(int_param(instr, "init_addr").value_or(0));
        continue;
      }

      // A rep or trans configures the AGU of the resource's last evt, and
      // needs that evt to have opened a lane of the configuration it is in.
      auto agu = current.find(resource->second);
      auto it = agu == current.end() ? pending.end() : pending.find(agu->second);
      if (it == pending.end()) {
        return instr.emitError("no-reconfiguration: ")
               << type << " before the evt of the configuration it is in";
      }
      if (type == "rep") {
        if (widths == rep_widths.end()) {
          return instr.emitError("no-reconfiguration: the ISA has no rep for ")
                 << entry.kind;
        }
        it->second.second.rep(int_param(instr, "ext").value_or(0) != 0,
                              int_param(instr, "iter").value_or(0),
                              int_param(instr, "step").value_or(0),
                              int_param(instr, "delay").value_or(0));
      } else {
        it->second.second.trans(int_param(instr, "delay").value_or(0));
      }
    }

    if (!length) {
      return raw.emitError("no-reconfiguration: cell (")
             << row << ", " << col << ") never halts";
    }

    // Configured but never started: it never runs, so there is nothing to
    // hardcode.
    for (const auto &[key, entry] : pending) {
      InstrOp evt = entry.first;
      evt.emitWarning("no-reconfiguration: this AGU configuration is "
                              "never activated and is dropped");
    }

    lengths_[{row, col}] = *length;
    return mlir::success();
  }

  // Record a conf entry of `resource`. An entry of a kind without options
  // holds one value for the whole program, so writing it again with a
  // different value fails. An entry of an option table that is written again
  // with a different value starts a new version of its option instead, which
  // allocate_options later gives an option of its own.
  mlir::LogicalResult
  record_conf(InstrOp instr, uint64_t cycle, const ResourceEntry &resource,
              const std::map<std::pair<std::string, std::string>, int>
                  &option_counts,
              ResourceConf &confs) {
    std::string variant = string_param(instr, "variant");
    const ConfRule *rule = nullptr;
    for (const ConfRule &candidate : conf_rules()) {
      if (resource.kind == candidate.kind && variant == candidate.variant) {
        rule = &candidate;
        break;
      }
    }
    if (!rule) {
      return instr.emitError("no-reconfiguration: cannot hardcode a conf")
             << (variant.empty() ? "" : " " + variant) << " of a "
             << resource.kind;
    }

    nlohmann::json fields = params_json(instr);
    nlohmann::json key = {{"variant", variant}};
    nlohmann::json value = nlohmann::json::object();
    for (const std::string &name : rule->key) {
      key[name] = fields.value(name, nlohmann::json());
    }
    for (const std::string &name : rule->value) {
      value[name] = fields.value(name, nlohmann::json());
    }
    auto value_of = [&](const nlohmann::json &written) {
      nlohmann::json written_value = nlohmann::json::object();
      for (const std::string &name : rule->value) {
        written_value[name] = written.value(name, nlohmann::json());
      }
      return written_value;
    };

    if (rule->selector_port >= 0) {
      auto count = option_counts.find({resource.kind, variant});
      if (count == option_counts.end()) {
        return instr.emitError("no-reconfiguration: the ISA has no option "
                               "field for this conf");
      }
      OptionTable &table = confs.tables[rule->table];
      table.rule = rule;
      table.num_options = count->second;

      int option = int_param(instr, "option").value_or(0);
      std::vector<OptionVersion> &versions = table.options[option];
      auto start_version = [&](std::map<std::string, nlohmann::json> entries) {
        OptionVersion version;
        version.entries = std::move(entries);
        version.first = instr;
        version.cycle = cycle;
        version.held = table.held == option;
        versions.push_back(std::move(version));
      };
      if (versions.empty()) {
        start_version({});
      }
      // An activation that read the version must not see what a later conf
      // writes, so any change after it starts a new version; before it, only
      // rewriting an entry this version wrote itself does.
      OptionVersion &current = versions.back();
      auto written = current.entries.find(key.dump());
      bool added = written == current.entries.end();
      bool rewritten = !added && value_of(written->second) != value;
      if ((current.activated && (added || rewritten)) ||
          (rewritten && current.written.count(key.dump()))) {
        start_version(current.entries);
      }
      versions.back().entries[key.dump()] = fields;
      versions.back().written.insert(key.dump());
      return mlir::success();
    }

    auto [it, inserted] = confs.fixed.insert({key.dump(), {instr, fields}});
    if (inserted) {
      return mlir::success();
    }
    nlohmann::json earlier = value_of(it->second.second);
    if (earlier != value) {
      mlir::InFlightDiagnostic diag = instr.emitError(
          "no-reconfiguration: conf rewrites an entry of the ")
          << resource.kind << " in slot " << resource.slot << " of cell ("
          << resource.row << ", " << resource.col << ") with " << value.dump()
          << ", already written with " << earlier.dump();
      diag.attachNote(it->second.first.getLoc()) << "first written here";
      return mlir::failure();
    }
    return mlir::success();
  }

  // Give every version of every option of `table` an option of its own for
  // the whole program, so that no conf has to rewrite one, and start each
  // activation of the selector at the option its first version got.
  //
  // The resource selects option 0 from reset, so the first version of option
  // 0 keeps option 0 if the resource uses it before any activation selects
  // another. In program order, each activation gets the lowest base option
  // from which every address it reads lands on an option that is free or
  // already holds the same entries; its configuration's initial addresses are
  // offset by that base.
  //
  // A version no activation reads, which the resource uses because the
  // option it selects was rewritten, is switched to by an activation added
  // for it: one, for all such versions of the table, whose configuration has
  // one lane per version, addressing the option it got, and a transition
  // delay between lanes that presents each address in the cycle the rewrite
  // took effect. That is only possible for a selector the program never
  // activates itself, since the added run keeps it busy until the last
  // switch. `configs` are the configurations of the resource's AGUs and
  // `activations` those of its cell, by cycle.
  mlir::LogicalResult allocate_options(
      OptionTable &table, const ResourceEntry &resource,
      std::map<std::pair<int, int>, nlohmann::json> &configs,
      std::map<uint64_t, std::map<int, int>> &activations) {
    using Entries = std::map<std::string, nlohmann::json>;
    table.slots.assign(table.num_options, std::nullopt);
    std::pair<int, int> selector{resource.slot, table.rule->selector_port};

    struct Switch {
      uint64_t cycle;
      const Entries *entries;
      InstrOp conf;
    };
    std::vector<Switch> switches;
    for (auto &[option, versions] : table.options) {
      for (size_t v = 0; v < versions.size(); v++) {
        OptionVersion &version = versions[v];
        if (version.activated) {
          continue;
        }
        if (!version.held) {
          version.first.emitWarning("no-reconfiguration: no activation reads "
                                    "this ")
              << table.rule->table << " option, so it is dropped";
          continue;
        }
        if (option == 0 && v == 0) {
          table.slots[0] = version.entries;
          continue;
        }
        switches.push_back({version.cycle, &version.entries, version.first});
      }
    }
    std::sort(switches.begin(), switches.end(),
              [](const Switch &a, const Switch &b) { return a.cycle < b.cycle; });

    auto selector_configs = configs.find(selector);
    if (!switches.empty() && selector_configs != configs.end() &&
        !selector_configs->second.empty()) {
      return switches.front().conf.emitError("no-reconfiguration: this conf "
                                             "rewrites ")
             << table.rule->table << " option of the " << resource.kind
             << " in slot " << resource.slot
             << " while the resource selects it, and the program also "
                "activates the AGU that selects it; switching options both "
                "ways cannot be hardcoded";
    }

    // The lowest option from `span` options on that can hold `reads`.
    auto place = [&](const std::map<uint64_t, Entries> &reads)
        -> std::optional<uint64_t> {
      uint64_t span = reads.rbegin()->first;
      for (uint64_t b = 0; b + span < static_cast<uint64_t>(table.num_options);
           b++) {
        bool fits = true;
        for (const auto &[addr, entries] : reads) {
          const auto &slot = table.slots[b + addr];
          if (slot && *slot != entries) {
            fits = false;
            break;
          }
        }
        if (fits) {
          for (const auto &[addr, entries] : reads) {
            table.slots[b + addr] = entries;
          }
          return b;
        }
      }
      return std::nullopt;
    };

    for (OptionUse &use : table.uses) {
      std::map<uint64_t, Entries> reads;
      for (const auto &[addr, read] : use.reads) {
        reads[addr] = read.second
                          ? table.options[read.first][*read.second].entries
                          : Entries{};
      }
      auto base = place(reads);
      if (!base) {
        return use.act.emitError("no-reconfiguration: the ")
               << table.rule->table << " options this act starts reading do "
               << "not fit in the " << table.num_options
               << " options of the resource next to the ones earlier "
                  "activations read";
      }
      if (*base != 0) {
        nlohmann::json &lanes = configs[selector][use.config]["lanes"];
        if (lanes.empty()) {
          return use.act.emitError("no-reconfiguration: this act starts an "
                                   "unconfigured AGU, which cannot start at "
                                   "option ")
                 << *base;
        }
        for (nlohmann::json &lane : lanes) {
          lane["init_addr"] = lane["init_addr"].get<uint64_t>() + *base;
        }
      }
    }

    if (switches.empty()) {
      return mlir::success();
    }

    // A rewrite issued in cycle c takes effect in cycle c + 1; the added run
    // presents its address agu_latency cycles after its act, and the resource
    // switches select_latency cycles after that.
    int64_t act_cycle = static_cast<int64_t>(switches.front().cycle) + 1 -
                        resource.agu_latency - table.rule->select_latency;
    if (act_cycle < 0) {
      return switches.front().conf.emitError("no-reconfiguration: this ")
             << table.rule->table << " option takes effect too early to be "
                "switched to by an activation";
    }
    nlohmann::json lanes = nlohmann::json::array();
    nlohmann::json mt = nlohmann::json::array();
    for (size_t k = 0; k < switches.size(); k++) {
      auto option = place({{0, *switches[k].entries}});
      if (!option) {
        return switches[k].conf.emitError("no-reconfiguration: the ")
               << table.rule->table << " option this conf rewrites does not "
               << "fit in the " << table.num_options
               << " options of the resource";
      }
      lanes.push_back(
          {{"init_addr", *option},
           {"ir", nlohmann::json::array(
                      {{{"iter", 1}, {"step", 0}, {"delay", 0}}})}});
      if (k > 0) {
        mt.push_back(
            {{"delay", switches[k].cycle - switches[k - 1].cycle - 1}});
      }
    }
    configs[selector].push_back(
        {{"lanes", lanes}, {"mt", mt}, {"or", nlohmann::json::array()}});
    activations[act_cycle][resource.slot] |= 1 << table.rule->selector_port;
    return mlir::success();
  }
};

} // namespace
} // namespace vesyla::pasm
