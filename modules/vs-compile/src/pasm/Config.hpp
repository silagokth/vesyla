#ifndef __VESYLA_PASM_CONFIG_HPP__
#define __VESYLA_PASM_CONFIG_HPP__

#include "util/Common.hpp"
#include <string>

namespace vesyla {
namespace pasm {

// create a singlton class for configuration storage. It must be a static class
class Config {
public:
  void set_isa_json(std::string isa_json_path);
  void set_arch_json(std::string arch_json_path);
  nlohmann::json get_arch_json() const;
  nlohmann::json get_isa_json() const;
  nlohmann::json get_component_map_json() const;

  // architecture json
  static nlohmann::json arch_json;
  // isa json
  static nlohmann::json isa_json;
  // component map json
  static nlohmann::json component_map_json;
};

} // namespace pasm
} // namespace vesyla

#endif // __VESYLA_PASM_CONFIG_HPP__