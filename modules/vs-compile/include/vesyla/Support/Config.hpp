#ifndef __VESYLA_PASM_CONFIG_HPP__
#define __VESYLA_PASM_CONFIG_HPP__

#include "vesyla/Support/Common.hpp"
#include <string>

namespace vesyla {
namespace pasm {

// Direction and data kind carried by a port, looked up by port number.
struct PortInfo {
  // "input" or "output"
  std::string dir;
  // "word" or "bulk"
  std::string kind;
};

// create a singlton class for configuration storage. It must be a static class
class Config {
public:
  void set_isa_json(std::string isa_json_path);
  void set_arch_json(std::string arch_json_path);
  // Load the top-level vs-compile config. Currently it holds the "port_table"
  // section (per-port direction and data kind).
  void set_config_json(std::string config_json_path);
  nlohmann::json get_arch_json() const;
  nlohmann::json get_isa_json() const;
  nlohmann::json get_component_map_json() const;
  nlohmann::json get_config_json() const;

  // Look up the direction/kind for a port number from the config's
  // "port_table" section. Falls back to the default mapping (bit0 = direction,
  // bit1 = kind) when no config was loaded.
  PortInfo get_port_info(int port) const;

  // Resolve an output path or filename token from the config's "output"
  // section. The returned value is relative to the compile output directory
  // (the --output DIR). "${key}" references to other "output" entries are
  // substituted, so composed paths (e.g. "${debug_dir}/compile") stay in the
  // config. Any key the config does not override falls back to a built-in
  // default; an unknown key yields "".
  std::string output_path(const std::string &key) const;
  // Resolve a helper-script path from the config's "scripts" section. The
  // returned value is relative to the program directory. Falls back to a
  // built-in default.
  std::string script_path(const std::string &key) const;
  // Resolve an external tool name/path from the config's "tools" section.
  // Falls back to a built-in default.
  std::string tool_name(const std::string &key) const;

  // architecture json
  static nlohmann::json arch_json;
  // isa json
  static nlohmann::json isa_json;
  // component map json
  static nlohmann::json component_map_json;
  // top-level vs-compile config json. Currently holds the "port_table" section
  // (per-port direction and data kind).
  static nlohmann::json config_json;
};

} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_CONFIG_HPP__