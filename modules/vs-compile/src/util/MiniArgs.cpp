#include "MiniArgs.hpp"

namespace vesyla {
namespace util {
bool MiniArgs::parse(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      auto eq = arg.find('=');
      if (eq != std::string::npos) {
        values[arg.substr(2, eq - 2)] = arg.substr(eq + 1);
      } else if (i + 1 < argc && argv[i + 1][0] != '-') {
        values[arg.substr(2)] = argv[++i];
      } else {
        flags[arg.substr(2)] = true;
      }
    } else if (arg[0] == '-' && arg.size() == 2) {
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        values[arg.substr(1)] = argv[++i];
      } else {
        flags[arg.substr(1)] = true;
      }
    }
  }
  return true;
}

std::string MiniArgs::get(const std::string &key,
                          const std::string &def) const {
  auto it = values.find(key);
  return (it != values.end()) ? it->second : def;
}

bool MiniArgs::flag(const std::string &key) const {
  return flags.count(key) > 0;
}

bool MiniArgs::has(const std::string &key) const {
  return values.count(key) > 0;
}
} // namespace util
} // namespace vesyla