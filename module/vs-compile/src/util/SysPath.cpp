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

namespace vesyla {
namespace util {

SysPath::SysPath() {
  GlobalVar glv;

  glv.puts("$CURR_DIR", curr_dir());
  glv.puts("$HOME_DIR", home_dir());
  glv.puts("$TEMP_DIR", temp_dir());
  glv.puts("$PROG_DIR", prog_dir());
}

string SysPath::find_config_file(const string filename) {
  GlobalVar glv;
  string config_path;
  string config_file;
  string vs_module;
  if (!glv.gets("$VS_MODULE", vs_module)) {
    return "";
  }
  glv.gets("$CURR_DIR", config_path);
  config_file = config_path + filename;
  if (boost::filesystem::exists(config_file)) {
    return config_file;
  } else {
    glv.gets("$HOME_DIR", config_path);
    config_file = config_path + ".config/vesyla-suite/" + vs_module +
                  "/config/" + filename;
    if (boost::filesystem::exists(config_file)) {
      return config_file;
    } else {
      glv.gets("$PROG_DIR", config_path);
      config_file = config_path + "share/vesyla-suite/" + vs_module +
                    "/config/" + filename;
      if (boost::filesystem::exists(config_file)) {
        return config_file;
      } else {
        return "";
      }
    }
  }
}

string SysPath::curr_dir() {
  boost::filesystem::path full_path(boost::filesystem::current_path());
  return full_path.string() + "/";
}

string SysPath::home_dir() {
  boost::filesystem::path full_path(getenv("HOME"));
  return full_path.string() + "/";
}

string SysPath::temp_dir() {
  boost::filesystem::path full_path("/tmp/");
  return full_path.string();
}

string SysPath::prog_dir() {
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
  boost::filesystem::path full_path(path);
  full_path =
      boost::filesystem::canonical(full_path.parent_path().parent_path());
  return full_path.string() + "/";
}

} // namespace util
} // namespace vesyla

