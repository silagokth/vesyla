// miniargs.hpp
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace vesyla {
namespace util {
class MiniArgs {
private:
  std::unordered_map<std::string, std::string> values;
  std::unordered_map<std::string, bool> flags;

public:
  bool parse(int argc, char **argv);
  std::string get(const std::string &key, const std::string &def = "") const;
  bool flag(const std::string &key) const;
  bool has(const std::string &key) const;
};
} // namespace util
} // namespace vesyla
