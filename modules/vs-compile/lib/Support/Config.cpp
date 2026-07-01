#include "vesyla/Support/Config.hpp"

namespace vesyla {
namespace pasm {

nlohmann::json Config::isa_json;
nlohmann::json Config::arch_json;
nlohmann::json Config::component_map_json;
nlohmann::json Config::config_json;

using namespace std;

namespace {

// Built-in defaults for the config sections that describe where vs-compile
// writes its output. Keeping the defaults here means config.json only needs to
// list the keys a user wants to override; every unset key falls back to these.
// "${key}" values compose one entry from another and are resolved on lookup.
const std::map<std::string, std::string> &output_defaults() {
  static const std::map<std::string, std::string> defaults = {
      {"debug_dir", "debug"},
      {"compile_debug_dir", "${debug_dir}/compile"},
      {"schedule_debug_dir", "${debug_dir}/schedule"},
      {"vis_dir", "${debug_dir}/vis"},
      {"minizinc_dir", "${debug_dir}/minizinc"},
      {"compile_dir", "compile"},
      {"timetable_dir", "${compile_dir}/timetable"},
      {"interconnect_dir", "${debug_dir}/interconnect"},
      {"compile_stage_prefix", "scf_"},
      {"schedule_stage_prefix", ""},
      {"stage_ext", ".mlir"},
      {"instr_basename", "instr"},
      {"schedule_dump_prefix", "schedule_"},
  };
  return defaults;
}

// Helper scripts shipped next to the executable, relative to the program dir.
const std::map<std::string, std::string> &script_defaults() {
  static const std::map<std::string, std::string> defaults = {
      {"vis", "scripts/script.py"},
      {"vis_grouped", "scripts/script_grouped.py"},
      {"vis_grouped_no_slot0", "scripts/script_grouped_no_slot0.py"},
      {"timetable", "scripts/timetable.py"},
  };
  return defaults;
}

// External tools invoked by the compiler.
const std::map<std::string, std::string> &tool_defaults() {
  static const std::map<std::string, std::string> defaults = {
      {"compile_util", "compile_util"},
  };
  return defaults;
}

// Return the raw (unresolved) value for `key` in `section`: the config override
// if present and a string, otherwise the built-in default (or "" if the key is
// unknown to both).
std::string raw_value(const nlohmann::json &config_json,
                      const std::string &section, const std::string &key,
                      const std::map<std::string, std::string> &defaults) {
  if (config_json.is_object() && config_json.contains(section) &&
      config_json[section].is_object() && config_json[section].contains(key) &&
      config_json[section][key].is_string()) {
    return config_json[section][key].get<std::string>();
  }
  auto it = defaults.find(key);
  return it != defaults.end() ? it->second : std::string();
}

// Substitute "${key}" references in `value` with other entries of the same
// section (config override or default). The depth bound guards against a cyclic
// reference in a hand-edited config.
std::string resolve(const nlohmann::json &config_json,
                    const std::string &section, const std::string &value,
                    const std::map<std::string, std::string> &defaults,
                    int depth) {
  if (depth <= 0) {
    return value;
  }
  std::string result = value;
  std::string::size_type pos = 0;
  while ((pos = result.find("${", pos)) != std::string::npos) {
    std::string::size_type end = result.find('}', pos + 2);
    if (end == std::string::npos) {
      break;
    }
    std::string key = result.substr(pos + 2, end - (pos + 2));
    std::string replacement =
        resolve(config_json, section,
                raw_value(config_json, section, key, defaults), defaults,
                depth - 1);
    result.replace(pos, end - pos + 1, replacement);
    pos += replacement.size();
  }
  return result;
}

std::string lookup(const nlohmann::json &config_json, const std::string &section,
                   const std::string &key,
                   const std::map<std::string, std::string> &defaults) {
  return resolve(config_json, section,
                 raw_value(config_json, section, key, defaults), defaults, 16);
}

} // namespace

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

std::string Config::output_path(const std::string &key) const {
  return lookup(config_json, "output", key, output_defaults());
}
std::string Config::script_path(const std::string &key) const {
  return lookup(config_json, "scripts", key, script_defaults());
}
std::string Config::tool_name(const std::string &key) const {
  return lookup(config_json, "tools", key, tool_defaults());
}

} // namespace pasm
} // namespace vesyla
