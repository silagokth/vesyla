#include "Config.hpp"

namespace vesyla {
namespace pasm {

nlohmann::json Config::isa_json;
nlohmann::json Config::arch_json;
nlohmann::json Config::component_map_json;

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
        for (auto j = 0; j < 4; j++) {
          int port = j;
          std::string key = std::to_string(row) + "_" + std::to_string(col) +
                            "_" + std::to_string(slot) + "_" +
                            std::to_string(port);
          component_map_json[key] = resource_kind;
        }
      }
    }
  }
}
nlohmann::json Config::get_arch_json() const { return arch_json; }
nlohmann::json Config::get_isa_json() const { return isa_json; }
nlohmann::json Config::get_component_map_json() const {
  return component_map_json;
}

} // namespace pasm
} // namespace vesyla
