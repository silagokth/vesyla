// Copyright (C) 2022 Yu Yang
//
// This file is part of vesyla-suite.
//
// vesyla-suite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// vesyla-suite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with vesyla-suite.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __VS_UTIL_SYS_PATH_HPP__
#define __VS_UTIL_SYS_PATH_HPP__

#include "GlobalVar.hpp"
#include "RandName.hpp"
#include <filesystem>
#include <string>

using namespace std;

namespace vesyla {
namespace util {

class SysPath {
public:
  static string curr_dir();
  static string home_dir();
  static string temp_dir();
  static string prog_dir();

private:
  static void register_curr_dir();
  static void register_home_dir();
  static void register_temp_dir();
  static void register_prog_dir();
};

} // namespace util
} // namespace vesyla

#endif // __VS_UTIL_SYS_PATH_HPP__
