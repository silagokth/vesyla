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

#ifndef __VESYLA_PASMPAR_FLEX_UTIL_HPP__
#define __VESYLA_PASMPAR_FLEX_UTIL_HPP__

#include "global_util.hpp"
#include "util/Common.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>

using std::cout;
using std::string;

namespace vesyla {
namespace pasmpar {
void print_lex_token(const string &token_);
} // namespace pasmpar
} // namespace vesyla

#endif // __VESYLA_PASMPAR_FLEX_UTIL_HPP__
