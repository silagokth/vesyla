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
#include "inja/inja.hpp"
#include "json/json.hpp"
#include <cfloat>
#include <limits.h>
#include <map>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "Config.hpp"
#include "SysPath.hpp"

#define BOOST_ALLOW_DEPRECATED_HEADERS
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

using json = nlohmann::json;

#define __NOT_IMPLEMENTED__                                                    \
  BOOST_LOG_TRIVIAL(fatal) << "Function has not been implemented yet!";        \
  std::exit(-1);
#define __NOT_SUPPORTED__                                                      \
  BOOST_LOG_TRIVIAL(fatal) << "Function is not supported!";
#define __DEPRECATED__                                                         \
  LOG(WARNING) << "Function is deprecated and will be removed soon!";
#define __VIRTUAL_FUNCTION__                                                   \
  BOOST_LOG_TRIVIAL(fatal) << "Virtual function cannot be directly accessed!"; \
  std::exit(-1);

class Common {
public:
  // __BLANK__;
};

#endif // __VESYLA_UTIL_COMMON_HPP__
