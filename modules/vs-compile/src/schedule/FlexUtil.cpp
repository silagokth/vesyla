
#include "FlexUtil.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>

using std::cout;
using std::string;

namespace vesyla {
namespace schedule {

void print_lex_token(const string &token_) {
  LOG_DEBUG << "FLEX: Source line: " << global_source_line
            << ", returned token: " << token_;
}

} // namespace schedule
} // namespace vesyla
