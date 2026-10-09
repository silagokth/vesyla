#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include "vesyla/Dialect/Pasm/Transforms/SizeAguPass.hpp"
#include "vesyla/Support/Config.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace vesyla::pasm {
#define GEN_PASS_DEF_SIZEAGUPASS
#include "vesyla/Dialect/Pasm/Transforms/Passes.hpp.inc"

namespace {

// Ports per slot, the way `act` and the component map number them: port index
// `slot * 4 + port`.
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
  std::vector<std::string> parameters;
};

// The AGU parameters this pass sizes. They start at what an AGU the program
// never configures gets: the smallest one agu_rtr elaborates, which needs at
// least one IR level.
struct AguSize {
  uint64_t number_ir = 1;
  uint64_t number_mt = 0;
  uint64_t number_or = 0;

  void merge(const AguSize &other) {
    number_ir = std::max(number_ir, other.number_ir);
    number_mt = std::max(number_mt, other.number_mt);
    number_or = std::max(number_or, other.number_or);
  }
};

// One configuration of one AGU, replayed the way agu_controller decodes it.
//
// An `evt` opens the next lane and restarts the level count. A base `rep`
// (ext = 0) takes the next level: an IR level of the current lane until a
// `trans` has been seen, an OR level after it; its `repx` (ext = 1) fills the
// upper half of the same level and takes none. A `trans` takes the next MT
// entry and sends every later `rep` to OR, which a later `evt` does not undo;
// it is dropped once an OR level was written. The IR levels are sized to the
// deepest lane, so none of them spills into OR the way the controller spills a
// lane deeper than NUMBER_IR.
class AguReplay {
public:
  void evt() {
    lanes_++;
    level_ = 0;
  }

  void rep(bool ext) {
    if (ext) {
      return;
    }
    if (use_or_) {
      or_levels_ = std::max(or_levels_, level_ + 1);
      or_configured_ = true;
    } else {
      ir_levels_ = std::max(ir_levels_, level_ + 1);
    }
    level_++;
  }

  void trans() {
    if (or_configured_) {
      return;
    }
    transitions_++;
    use_or_ = true;
    level_ = 0;
  }

  // An AGU no `evt` opened was never configured.
  bool configured() const { return lanes_ > 0; }

  // The lanes are indexed 0..NUMBER_MT and the transitions 0..NUMBER_MT-1, so
  // NUMBER_MT has to cover both. An `evt` alone already configures IR level 0
  // of its lane.
  AguSize size() const {
    AguSize size;
    size.number_ir = std::max<uint64_t>(ir_levels_, 1);
    size.number_mt = std::max(lanes_ - 1, transitions_);
    size.number_or = or_levels_;
    return size;
  }

private:
  uint64_t lanes_ = 0;
  uint64_t level_ = 0;
  uint64_t transitions_ = 0;
  uint64_t ir_levels_ = 0;
  uint64_t or_levels_ = 0;
  bool use_or_ = false;
  bool or_configured_ = false;
};

std::optional<int64_t> int_param(InstrOp instr, llvm::StringRef name) {
  auto attr = instr.getParam().getAs<mlir::IntegerAttr>(name);
  if (!attr) {
    return std::nullopt;
  }
  return attr.getInt();
}

// The (slot, port) of every AGU an `act` starts, decoded per mode the way
// InstrFactory encodes them. Nothing for a mode this does not know.
std::optional<std::vector<std::pair<int, int>>> activated_ports(InstrOp act) {
  auto mode = int_param(act, "mode");
  auto param = int_param(act, "param");
  auto ports = int_param(act, "ports");
  if (!mode || !param || !ports) {
    return std::nullopt;
  }

  uint64_t mask = static_cast<uint64_t>(*ports);
  std::vector<int> indices;
  if (*mode == 0) {
    // `ports` counts port indices from the first port of slot `param`.
    for (int bit = 0; bit < 64; bit++) {
      if (mask & (1ULL << bit)) {
        indices.push_back(*param * ports_per_slot + bit);
      }
    }
  } else if (*mode == 1) {
    // `ports` is a mask of slots, `param` a mask of the ports in each.
    for (int slot = 0; slot < 64; slot++) {
      if (!(mask & (1ULL << slot))) {
        continue;
      }
      for (int port = 0; port < ports_per_slot; port++) {
        if (*param & (1 << port)) {
          indices.push_back(slot * ports_per_slot + port);
        }
      }
    }
  } else if (*mode == 2) {
    // `ports` is the port map the sequencer register is loaded with.
    for (int bit = 0; bit < 64; bit++) {
      if (mask & (1ULL << bit)) {
        indices.push_back(bit);
      }
    }
  } else {
    return std::nullopt;
  }

  std::vector<std::pair<int, int>> activated;
  for (int index : indices) {
    activated.push_back({index / ports_per_slot, index % ports_per_slot});
  }
  return activated;
}

class SizeAguPass : public impl::SizeAguPassBase<SizeAguPass> {
public:
  using impl::SizeAguPassBase<SizeAguPass>::SizeAguPassBase;

  void runOnOperation() final {
    Config cfg;
    nlohmann::json arch = cfg.get_arch_json();

    // Every resource in the architecture, and which one occupies each slot.
    // A slot no resource occupies is in neither, and anything addressed to it
    // is ignored.
    std::vector<ResourceEntry> resources;
    std::map<SlotKey, size_t> resource_at;
    for (const auto &cell : arch["cells"]) {
      int row = cell["coordinates"]["row"];
      int col = cell["coordinates"]["col"];
      for (const auto &resource : cell["cell"]["resources_list"]) {
        int first_slot = resource["slot"];
        int size = resource["size"];
        ResourceEntry entry{row, col, first_slot, {}};
        if (resource.contains("parameters")) {
          for (const auto &parameter : resource["parameters"].items()) {
            entry.parameters.push_back(parameter.key());
          }
        }
        for (int slot = first_slot; slot < first_slot + size; slot++) {
          resource_at[{row, col, slot}] = resources.size();
        }
        resources.push_back(entry);
      }
    }
    std::vector<AguSize> sizes(resources.size());

    auto result = getOperation().walk([&](RawOp raw) {
      int row = raw.getRow();
      int col = raw.getCol();

      // The configuration each AGU has been given since it last started, and
      // per resource the AGU its last `evt` addressed: a `rep` or `trans`
      // configures that one, whichever port it names itself.
      std::map<AguKey, AguReplay> replays;
      std::map<size_t, AguKey> current;

      auto record = [&](const AguKey &key, const AguReplay &replay) {
        if (!replay.configured()) {
          return;
        }
        auto [agu_row, agu_col, slot, port] = key;
        sizes[resource_at.at({agu_row, agu_col, slot})].merge(replay.size());
      };

      for (InstrOp instr : raw.getBody().front().getOps<InstrOp>()) {
        llvm::StringRef type = instr.getType();

        // An AGU runs the configuration it was given when it is started, and
        // comes back from the run cleared.
        if (type == "act") {
          auto activated = activated_ports(instr);
          if (!activated) {
            instr.emitError("size-agu-pass: cannot tell which ports this act "
                            "starts");
            return mlir::WalkResult::interrupt();
          }
          for (auto [slot, port] : *activated) {
            auto it = replays.find({row, col, slot, port});
            if (it != replays.end()) {
              record(it->first, it->second);
              replays.erase(it);
            }
          }
          continue;
        }

        if (type != "evt" && type != "rep" && type != "trans") {
          continue;
        }
        auto slot = int_param(instr, "slot");
        if (!slot) {
          continue;
        }
        auto resource = resource_at.find({row, col, static_cast<int>(*slot)});
        if (resource == resource_at.end()) {
          continue;
        }

        if (type == "evt") {
          auto port = int_param(instr, "port");
          if (!port) {
            continue;
          }
          AguKey key{row, col, static_cast<int>(*slot),
                     static_cast<int>(*port)};
          current[resource->second] = key;
          replays[key].evt();
          continue;
        }

        // Before any `evt` there is no AGU for it to configure.
        auto agu = current.find(resource->second);
        if (agu == current.end()) {
          continue;
        }
        AguReplay &replay = replays[agu->second];
        if (type == "rep") {
          replay.rep(int_param(instr, "ext").value_or(0) != 0);
        } else {
          replay.trans();
        }
      }

      // Configured but never started in this stream: still what it is sized
      // for.
      for (const auto &[key, replay] : replays) {
        record(key, replay);
      }
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Only a parameter the resource already has is set, so a kind of resource
    // without AGUs is left as it is.
    for (size_t i = 0; i < resources.size(); i++) {
      const ResourceEntry &resource = resources[i];
      const std::pair<const char *, uint64_t> values[] = {
          {"NUMBER_IR", sizes[i].number_ir},
          {"NUMBER_MT", sizes[i].number_mt},
          {"NUMBER_OR", sizes[i].number_or},
      };
      for (const auto &[name, value] : values) {
        if (std::find(resource.parameters.begin(), resource.parameters.end(),
                      name) == resource.parameters.end()) {
          continue;
        }
        cfg.set_resource_parameter(resource.row, resource.col, resource.slot,
                                   name, value);
      }
    }
  }
};

} // namespace
} // namespace vesyla::pasm
