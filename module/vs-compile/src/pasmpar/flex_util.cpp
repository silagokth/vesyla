
#include "flex_util.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>

using std::cout;
using std::string;

namespace vesyla {
namespace pasmpar {

void print_lex_token(const string &token_) {
  LOG(DEBUG) << "FLEX: Source line: " << global_source_line
             << ", returned token: " << token_;
}

} // namespace pasmpar
} // namespace vesyla
