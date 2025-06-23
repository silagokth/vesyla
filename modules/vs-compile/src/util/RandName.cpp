#include "RandName.hpp"

namespace vesyla {
namespace util {
std::string RandName::generate(size_t length) {
  if (length == 0) {
    return "";
  }

  // generate a random string starting with a alphabetic character
  const std::string letters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const std::string numbers = "0123456789";
  const std::string characters = letters + numbers;
  std::string result;
  result.reserve(length);
  result += letters[rand() % letters.size()];
  for (size_t i = 1; i < length; ++i) {
    result += characters[rand() % characters.size()];
  }
  return result;
}

} // namespace util
} // namespace vesyla