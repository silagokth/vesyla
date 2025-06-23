// Copyright (C) 2019 Yu Yang
//
// This file is part of Vesyla.
//
// Vesyla is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Vesyla is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Vesyla.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __VESYLA_UTIL_COMMON_HPP__
#define __VESYLA_UTIL_COMMON_HPP__

#include "GlobalVar.hpp"
#include "MiniArgs.hpp"
#include "RandName.hpp"
#include "SysPath.hpp"
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Log.h"
#include "json/json.hpp"
#include <cfloat>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <map>
#include <math.h>
#include <set>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define BOOST_ALLOW_DEPRECATED_HEADERS
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#define __NOT_IMPLEMENTED__                                                    \
  LOG(FATAL) << "Function has not been implemented yet!";                      \
  std::exit(-1);
#define __NOT_SUPPORTED__ LOG(FATAL) << "Function is not supported!";
#define __DEPRECATED__                                                         \
  LOG(WARNING) << "Function is deprecated and will be removed soon!";
#define __VIRTUAL_FUNCTION__                                                   \
  LOG(FATAL) << "Virtual function cannot be directly accessed!";               \
  std::exit(-1);

namespace vesyla {
namespace util {

class Common {
public:
  static std::string gen_random_string(size_t length) {
    return RandName::generate(length);
  }

  static std::string remove_leading_and_trailing_white_space(std::string line) {
    // remove the leading and trailing spaces
    const char *WhiteSpace = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(WhiteSpace);
    std::size_t end = line.find_last_not_of(WhiteSpace);
    line = start == end ? std::string() : line.substr(start, end - start + 1);
    return line;
  }
};
} // namespace util
} // namespace vesyla

#endif // __VESYLA_UTIL_COMMON_HPP__
