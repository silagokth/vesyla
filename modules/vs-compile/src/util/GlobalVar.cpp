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

#include "GlobalVar.hpp"

namespace vesyla {
namespace util {

map<string, GlobalVar::glvar_t> GlobalVar::_pool;

bool GlobalVar::get(const string key, glvar_t &val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  if (it != _pool.end()) {
    val = it->second;
    return true;
  } else {
    return false;
  }
}

bool GlobalVar::getb(const string key, bool &val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  if (it != _pool.end()) {
    if (it->second.type != "boolean") {
      LOG_FATAL << "Type wrong: global variable \"" << key << "\" is "
                << it->second.type;
    }
    val = it->second.b;
    return true;
  } else {
    return false;
  }
}

bool GlobalVar::geti(const string key, int &val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  if (it != _pool.end()) {
    if (it->second.type != "integer") {
      LOG_FATAL << plog::error << "Type wrong: global variable \"" << key
                << "\" is " << it->second.type;
    }
    val = it->second.i;
    return true;
  } else {
    return false;
  }
}

bool GlobalVar::getf(const string key, float &val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  if (it != _pool.end()) {
    if (it->second.type != "float") {
      LOG_FATAL << plog::error << "Type wrong: global variable \"" << key
                << "\" is " << it->second.type;
    }
    val = it->second.f;
    return true;
  } else {
    return false;
  }
}

bool GlobalVar::gets(const string key, string &val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  if (it != _pool.end()) {
    if (it->second.type != "string") {
      LOG_FATAL << plog::error << "Type wrong: global variable \"" << key
                << "\" is " << it->second.type;
    }
    val = it->second.s;
    return true;
  } else {
    return false;
  }
}

bool GlobalVar::put(const string key, const glvar_t val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  _pool[key] = val;
  return true;
}

bool GlobalVar::putb(const string key, const bool val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  glvar_t v;
  v.type = "boolean";
  v.b = val;
  _pool[key] = v;
  return true;
}

bool GlobalVar::puti(const string key, const int val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  glvar_t v;
  v.type = "integer";
  v.i = val;
  _pool[key] = v;
  return true;
}

bool GlobalVar::putf(const string key, const float val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  glvar_t v;
  v.type = "float";
  v.f = val;
  _pool[key] = v;
  return true;
}

bool GlobalVar::puts(const string key, const string val) {
  std::map<string, glvar_t>::iterator it;
  it = _pool.find(key);
  glvar_t v;
  v.type = "string";
  v.s = val;
  _pool[key] = v;
  return true;
}

bool GlobalVar::load_vars(const string filename_) {
  namespace fs = std::filesystem;

  std::ifstream inp(filename_);
  nlohmann::json config = nlohmann::json::parse(inp);
  bool flag = true;
  for (auto j : config) {
    for (auto i : j["variables"]) {
      if (i["type"] == "integer") {
        flag = puti((fs::path(j["path"].get<string>()) /
                     fs::path(i["name"].get<string>())),
                    i["value"]);
      } else if (i["type"] == "float") {
        flag = putf((fs::path(j["path"].get<string>()) /
                     fs::path(i["name"].get<string>())),
                    i["value"].get<float>());
      } else if (i["type"] == "string") {
        flag = puts((fs::path(j["path"].get<string>()) /
                     fs::path(i["name"].get<string>())),
                    i["value"]);
      } else if (i["type"] == "boolean") {
        flag = putb((fs::path(j["path"].get<string>()) /
                     fs::path(i["name"].get<string>())),
                    i["value"].get<bool>());
      } else {
        LOG_FATAL << "Type error: " << i["type"];
      };
    }
  }
  return true;
}

bool GlobalVar::store_vars(const string filename_) {
  namespace fs = std::filesystem;

  std::ofstream ofs(filename_, ofstream::out);
  nlohmann::json config;

  std::unordered_map<string, vector<string>> group;
  for (auto x : _pool) {
    fs::path p(x.first);
    if (group.find(p.parent_path()) == group.end()) {
      group[p.parent_path()] = {};
    }
    group[p.parent_path()].push_back(x.first);
  }

  for (auto x : group) {
    nlohmann::json j;
    j["path"] = x.first;
    nlohmann::json k;
    for (auto y : x.second) {
      nlohmann::json l;
      fs::path p(y);
      l["name"] = p.filename();
      l["type"] = _pool[y].type;
      bool flag;
      if (_pool[y].type == "integer") {
        int value;
        flag = geti(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "float") {
        float value;
        flag = getf(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "string") {
        string value;
        flag = gets(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "boolean") {
        bool value;
        flag = getb(y, value);
        l["value"] = value;
      } else {
        LOG_FATAL << "Wrong type: " << _pool[y].type;
      }
      if (flag)
        k.push_back(l);
    }
    j["variables"] = k;
    config.push_back(j);
  }

  ofs << config;
  ofs.close();

  return true;
}

bool GlobalVar::select_and_store_vars_white(const string filename_,
                                            std::set<string> filter_) {
  namespace fs = std::filesystem;

  std::ofstream ofs(filename_, ofstream::out);
  nlohmann::json config;

  std::unordered_map<string, vector<string>> group;
  for (auto x : _pool) {
    if (filter_.find(x.first) != filter_.end()) {
      fs::path p(x.first);
      if (group.find(p.parent_path()) == group.end()) {
        group[p.parent_path()] = {};
      }
      group[p.parent_path()].push_back(x.first);
    }
  }

  for (auto x : group) {
    nlohmann::json j;
    j["path"] = x.first;
    nlohmann::json k;
    for (auto y : x.second) {
      nlohmann::json l;
      fs::path p(y);
      l["name"] = p.filename();
      l["type"] = _pool[y].type;
      bool flag;
      if (_pool[y].type == "integer") {
        int value;
        flag = geti(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "float") {
        float value;
        flag = getf(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "string") {
        string value;
        flag = gets(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "boolean") {
        bool value;
        flag = getb(y, value);
        l["value"] = value;
      } else {
        LOG_FATAL << "Wrong type: " << _pool[y].type;
      }
      if (flag)
        k.push_back(l);
    }
    j["variables"] = k;
    config.push_back(j);
  }

  ofs << config;
  ofs.close();

  return true;
}

bool GlobalVar::select_and_store_vars_black(const string filename_,
                                            std::set<string> filter_) {
  namespace fs = std::filesystem;

  std::ofstream ofs(filename_, ofstream::out);
  nlohmann::json config;

  std::unordered_map<string, vector<string>> group;
  for (auto x : _pool) {
    if (filter_.find(x.first) == filter_.end()) {
      fs::path p(x.first);
      if (group.find(p.parent_path()) == group.end()) {
        group[p.parent_path()] = {};
      }
      group[p.parent_path()].push_back(x.first);
    }
  }

  for (auto x : group) {
    nlohmann::json j;
    j["path"] = x.first;
    nlohmann::json k;
    for (auto y : x.second) {
      nlohmann::json l;
      fs::path p(y);
      l["name"] = p.filename();
      l["type"] = _pool[y].type;
      bool flag;
      if (_pool[y].type == "integer") {
        int value;
        flag = geti(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "float") {
        float value;
        flag = getf(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "string") {
        string value;
        flag = gets(y, value);
        l["value"] = value;
      } else if (_pool[y].type == "boolean") {
        bool value;
        flag = getb(y, value);
        l["value"] = value;
      } else {
        LOG_FATAL << "Wrong type: " << _pool[y].type;
      }
      if (flag)
        k.push_back(l);
    }
    j["variables"] = k;
    config.push_back(j);
  }

  ofs << config;
  ofs.close();

  return true;
}

void GlobalVar::clear() { _pool.clear(); }

} // namespace util
} // namespace vesyla
