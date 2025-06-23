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

#include "SysPath.hpp"
#include "GlobalVar.hpp"

namespace vesyla {
namespace util {

string SysPath::curr_dir() {
  GlobalVar glv;
  string curr_dir;
  if (!glv.gets("__CURR_DIR__", curr_dir)) {
    register_curr_dir();
  }
  if (!glv.gets("__CURR_DIR__", curr_dir)) {
    LOG_FATAL << "Current directory is not set!";
    std::exit(-1);
  }
  return curr_dir;
}

string SysPath::home_dir() {
  GlobalVar glv;
  string home_dir;
  if (!glv.gets("__HOME_DIR__", home_dir)) {
    register_home_dir();
  }
  if (!glv.gets("__HOME_DIR__", home_dir)) {
    LOG_FATAL << "Home directory is not set!";
    std::exit(-1);
  }
  return home_dir;
}

string SysPath::temp_dir() {
  GlobalVar glv;
  string temp_dir;
  if (!glv.gets("__TEMP_DIR__", temp_dir)) {
    register_temp_dir();
  }
  if (!glv.gets("__TEMP_DIR__", temp_dir)) {
    LOG_FATAL << "Temporary directory is not set!";
    std::exit(-1);
  }
  return temp_dir;
}

string SysPath::prog_dir() {
  GlobalVar glv;
  string prog_dir;
  if (!glv.gets("__PROG_DIR__", prog_dir)) {
    register_prog_dir();
  }
  if (!glv.gets("__PROG_DIR__", prog_dir)) {
    LOG_FATAL << "Program directory is not set!";
    std::exit(-1);
  }
  return prog_dir;
}

void SysPath::register_curr_dir() {
  std::filesystem::path full_path(std::filesystem::current_path());
  GlobalVar glv;
  glv.puts("__CURR_DIR__", full_path.string() + "/");
}

void SysPath::register_home_dir() {
  std::filesystem::path full_path(getenv("HOME"));
  GlobalVar glv;
  glv.puts("__HOME_DIR__", full_path.string() + "/");
}

void SysPath::register_temp_dir() {
  string rn = RandName::generate(8);
  std::filesystem::path full_path("/tmp/vesyla_" + rn);
  while (std::filesystem::exists(full_path)) {
    rn = RandName::generate(8);
    full_path = std::filesystem::path("/tmp/vesyla_" + rn);
  }
  std::filesystem::create_directories(full_path);
  GlobalVar glv;
  glv.puts("__TEMP_DIR__", full_path.string() + "/");
}

void SysPath::register_prog_dir() {
  char szTmp[32];
  char pBuf[128];
  sprintf(szTmp, "/proc/%d/exe", getpid());
  int bytes = readlink(szTmp, pBuf, 128);
  if (bytes > 127) {
    bytes = 127;
  }
  if (bytes >= 0)
    pBuf[bytes] = '\0';
  string path(pBuf);
  std::filesystem::path full_path(path);
  full_path = std::filesystem::canonical(full_path.parent_path().parent_path());
  GlobalVar glv;
  glv.puts("__PROG_DIR__", full_path.string() + "/");
}

} // namespace util
} // namespace vesyla
