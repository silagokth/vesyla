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

  // Set a parameter of the resource that occupies `slot` in cell (row, col),
  // in the loaded architecture. Returns false when no resource occupies that
  // slot. The architecture write_arch_json writes carries every parameter set
  // this way; vesyla fabric generates the SST and RTL for a program from it,
  // taking the library default for any parameter left unset.
  bool set_resource_parameter(int row, int col, int slot,
                              const std::string &name, uint64_t value);
  // Set `instr_count` on the controller of cell (row, col), in the loaded
  // architecture: the number of instructions its sequencer is loaded with.
  // It is kept out of the controller's parameters, so it changes neither the
  // RTL nor the fingerprint vesyla fabric generates. Returns false when the
  // architecture has no such cell.
  bool set_controller_instr_count(int row, int col, uint64_t count);
  // Write the loaded architecture, with the parameters set through
  // set_resource_parameter and the counts set through
  // set_controller_instr_count, to `path`.
  void write_arch_json(const std::string &path) const;

  // Fields of the no-reconfiguration architecture (--no-reconfiguration).
  // They are kept apart from the loaded architecture, so arch_sized.json is
  // the same with or without the flag. Set `fields` on the controller of cell
  // (row, col), or on the resource that occupies `slot` in it; a later call
  // for the same target adds to or replaces earlier fields.
  void set_noreconf_controller_fields(int row, int col,
                                      const nlohmann::json &fields);
  void set_noreconf_resource_fields(int row, int col, int slot,
                                    const nlohmann::json &fields);
  // Write the loaded architecture, as write_arch_json does, with the fields
  // set above merged into its controllers and resources, to `path`. Returns
  // false when a field names a cell or slot the architecture does not have.
  bool write_noreconf_arch_json(const std::string &path) const;

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
  // Controller and resource fields of the no-reconfiguration architecture,
  // keyed "row_col" and "row_col_slot".
  static nlohmann::json noreconf_controllers;
  static nlohmann::json noreconf_resources;
};

} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_CONFIG_HPP__