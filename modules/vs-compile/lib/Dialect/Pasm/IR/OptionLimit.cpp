#include "vesyla/Dialect/Pasm/IR/OptionLimit.hpp"

#include "vesyla/Support/Config.hpp"
#include "vesyla/Support/GlobalVar.hpp"

#include <algorithm>
#include <map>
#include <string>

namespace vesyla::pasm {

namespace {

// Options the ISA option fields of each kind can name, as 1 << bits of the
// narrowest; kinds without such a field are absent.
const std::map<std::string, int64_t> &option_capacities() {
  static const std::map<std::string, int64_t> capacities = [] {
    std::map<std::string, int64_t> out;
    nlohmann::json isa = Config().get_isa_json();
    if (!isa.is_object() || !isa.contains("components")) {
      return out;
    }
    for (const auto &component : isa["components"]) {
      if (!component.is_object() || !component.contains("kind")) {
        continue;
      }
      std::string kind = component["kind"];
      auto scan = [&](const nlohmann::json &segments) {
        for (const auto &segment : segments) {
          std::string name = segment.value("name", "");
          if (name != "option" && name != "init_addr") {
            continue;
          }
          int64_t n = int64_t(1) << segment.value("bitwidth", 0);
          auto it = out.find(kind);
          out[kind] = it == out.end() ? n : std::min(it->second, n);
        }
      };
      for (const auto &instr :
           component.value("instructions", nlohmann::json::array())) {
        std::string name = instr.value("name", "");
        if (name != "conf" && name != "evt") {
          continue;
        }
        scan(instr.value("segments", nlohmann::json::array()));
        for (const auto &variant :
             instr.value("variants", nlohmann::json::array())) {
          scan(variant.value("segments", nlohmann::json::array()));
        }
      }
    }
    return out;
  }();
  return capacities;
}

} // namespace

int64_t option_limit(llvm::StringRef kind, int64_t num_options) {
  std::string flag;
  if (::vesyla::util::GlobalVar::gets("__NO_RECONFIGURATION__", flag) &&
      flag == "1") {
    return num_options;
  }
  const auto &capacity = option_capacities();
  auto it = capacity.find(kind.str());
  if (it == capacity.end() || num_options == 0) {
    return num_options;
  }
  return std::min(num_options, it->second);
}

int64_t option_limit_at(int row, int col, int slot) {
  nlohmann::json arch = Config().get_arch_json();
  if (!arch.is_object() || !arch.contains("cells")) {
    return option_limit("", 4);
  }
  for (const auto &entry : arch.at("cells")) {
    if (!entry.is_object() || !entry.contains("coordinates") ||
        !entry.contains("cell")) {
      continue;
    }
    const nlohmann::json &at = entry.at("coordinates");
    if (at.value("row", -1) != row || at.value("col", -1) != col) {
      continue;
    }
    const nlohmann::json &cell = entry.at("cell");
    if (!cell.is_object() || !cell.contains("resources_list")) {
      continue;
    }
    for (const auto &resource : cell.at("resources_list")) {
      if (resource.value("slot", -1) != slot) {
        continue;
      }
      int64_t n = 4;
      if (resource.contains("parameters")) {
        n = resource.at("parameters").value("NUM_OPTIONS", 4);
      }
      return option_limit(resource.value("kind", ""), n);
    }
  }
  return option_limit("", 4);
}

} // namespace vesyla::pasm
