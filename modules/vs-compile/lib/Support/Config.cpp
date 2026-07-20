#include "vesyla/Support/Config.hpp"

namespace vesyla {
namespace pasm {

nlohmann::json Config::isa_json;
nlohmann::json Config::arch_json;
nlohmann::json Config::component_map_json;
nlohmann::json Config::config_json;

using namespace std;

void Config::set_isa_json(std::string isa_json_path) {
  std::ifstream ifs(isa_json_path);
  if (!ifs.is_open()) {
    LOG_FATAL << "Error: Failed to open ISA JSON file: " << isa_json_path;
    std::exit(EXIT_FAILURE);
  }
  isa_json = nlohmann::json::parse(ifs);
  ifs.close();
}
void Config::set_arch_json(std::string arch_json_path) {
  std::ifstream ifs(arch_json_path);
  if (!ifs.is_open()) {
    LOG_FATAL << "Error: Failed to open Architecture JSON file: "
              << arch_json_path;
    std::exit(EXIT_FAILURE);
  }
  arch_json = nlohmann::json::parse(ifs);
  ifs.close();

  for (const auto &cell : arch_json["cells"].items()) {
    int row = cell.value()["coordinates"]["row"];
    int col = cell.value()["coordinates"]["col"];
    std::string controller_kind = cell.value()["cell"]["controller"]["kind"];
    std::string key = std::to_string(row) + "_" + std::to_string(col);
    component_map_json[key] = controller_kind;

    for (const auto &resource :
         cell.value()["cell"]["resources_list"].items()) {
      std::string resource_kind = resource.value()["kind"];
      int slot_start = resource.value()["slot"];
      for (auto i = 0; i < resource.value()["size"]; i++) {
        int slot = slot_start + i;
        // The resource kind is the same for every port of a slot, so it is
        // keyed by (row, col, slot). The per-port keys are kept too for any
        // caller that still resolves a resource by its full (row, col, slot,
        // port) location.
        std::string slot_key = std::to_string(row) + "_" + std::to_string(col) +
                               "_" + std::to_string(slot);
        component_map_json[slot_key] = resource_kind;
        for (auto j = 0; j < 4; j++) {
          int port = j;
          std::string key = slot_key + "_" + std::to_string(port);
          component_map_json[key] = resource_kind;
        }
      }
    }
  }
}
void Config::set_config_json(std::string config_json_path) {
  std::ifstream ifs(config_json_path);
  if (!ifs.is_open()) {
    LOG_FATAL << "Error: Failed to open config JSON file: " << config_json_path;
    std::exit(EXIT_FAILURE);
  }
  config_json = nlohmann::json::parse(ifs);
  ifs.close();
}
nlohmann::json Config::get_arch_json() const { return arch_json; }
nlohmann::json Config::get_isa_json() const { return isa_json; }
nlohmann::json Config::get_component_map_json() const {
  return component_map_json;
}
nlohmann::json Config::get_config_json() const { return config_json; }

PortInfo Config::get_port_info(int port) const {
  if (config_json.is_object() && config_json.contains("port_table") &&
      config_json["port_table"].is_array()) {
    for (const auto &entry : config_json["port_table"]) {
      if (entry.is_object() && entry.contains("port") &&
          entry["port"].is_number_integer() &&
          entry["port"].get<int>() == port) {
        std::string dir = entry.contains("dir") && entry["dir"].is_string()
                              ? entry["dir"].get<string>()
                              : "";
        std::string kind = entry.contains("kind") && entry["kind"].is_string()
                               ? entry["kind"].get<string>()
                               : "";
        return PortInfo{dir, kind};
      }
    }
  }
  // Fallback when no port table file was loaded: bit0 selects direction
  // (0 = input, 1 = output), bit1 selects data kind (0 = word, 1 = bulk).
  std::string dir = (port & 1) ? "output" : "input";
  std::string kind = (port & 2) ? "bulk" : "word";
  return PortInfo{dir, kind};
}

} // namespace pasm
} // namespace vesyla
