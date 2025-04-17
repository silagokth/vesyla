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

#include "Config.hpp"

namespace vesyla {
namespace util {

Config::Config(string config_file_) {
  using boost::property_tree::ptree;
  ptree pt;
  read_xml(config_file_, pt);
  GlobalVar glv;

  for (ptree::value_type &v : pt.get_child("config")) {
    if (v.first == "entry") {
      string name = v.second.get("<xmlattr>.name", "");
      string type = v.second.get("<xmlattr>.type", "");
      string value = v.second.get("<xmlattr>.value", "");
      if (type == "int") {
        glv.puti(name, stoi(value));
      } else if (type == "float") {
        glv.putf(name, stof(value));
      } else if (type == "bool") {
        glv.putb(name, stoi(value));
      } else {
        glv.puts(name, value);
      }
    }
  }
}

Config::~Config() {}

} // namespace util
} // namespace vesyla
