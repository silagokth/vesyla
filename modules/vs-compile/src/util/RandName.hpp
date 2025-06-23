#ifndef __VESYLA_UTIL_RAND_NAME_HPP__
#define __VESYLA_UTIL_RAND_NAME_HPP__

#include <chrono>
#include <cstdlib>
#include <string>

namespace vesyla {
namespace util {
class RandName {
public:
  static std::string generate(size_t length);
};
} // namespace util
} // namespace vesyla

#endif // __VESYLA_UTIL_RAND_NAME_HPP__