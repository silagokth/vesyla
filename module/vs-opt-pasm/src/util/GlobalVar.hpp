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

#ifndef __VS_UTIL_GLOBAL_VAR_HPP__
#define __VS_UTIL_GLOBAL_VAR_HPP__

#include "json/json.hpp"
#include <boost/log/trivial.hpp>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>

using namespace std;

namespace vesyla {
namespace util {

/**
 * \class GlobalVar
 * \brief The class manages all the global variables which supposed to be seen
 * by every other class.
 *
 * The GlobalVar maintains a variable pool of std::map type. Variables can be
 * accessed via a string type key. The variables are stored as a unified union
 * type called glvar_t. To easily access global variables of various original
 * types, GlobalVar class defines a set of API dealing with bool, int, float and
 * string type global variables.
 */
class GlobalVar {
public:
  /**
   * \struct glvar_t
   * \description The global variable entry type. It wraps all common variable
   * type to a single unified type.
   */
  struct glvar_t {
    string type; /**< type of the variable: boolean, integer, float, string */
    bool b;      /**< boolean type */
    int i;       /**< signed integer type */
    float f;     /**< single floating point type */
    char s[256]; /**< string type */
  };

  /**
   * \fn get()
   * \brief Search the glvar_t variable from the pool by a string type key
   * and return the result.
   * \param key The key to indicate the global variable.
   * \param val The glvar_t type container to receive the result.
   * \return True if found, otherwise false.
   */
  bool get(const string key, glvar_t &val);
  /**
   * \fn getb()
   * \brief Search the bool variable wrapped as glvar_t from the pool by a
   * string type key and return the result. \param key The key to indicate the
   * global variable. \param val The bool type container to receive the result.
   * \return True if found, otherwise false.
   */
  bool getb(const string key, bool &val);
  /**
   * \fn geti()
   * \brief Search the int variable wrapped as glvar_t from the pool by a string
   * type key and return the result. \param key The key to indicate the global
   * variable. \param val The int type container to receive the result. \return
   * True if found, otherwise false.
   */
  bool geti(const string key, int &val);
  /**
   * \fn getf()
   * \brief Search the float variable wrapped as glvar_t from the pool by a
   * string type key and return the result. \param key The key to indicate the
   * global variable. \param val The float type container to receive the result.
   * \return True if found, otherwise false.
   */
  bool getf(const string key, float &val);
  /**
   * \fn gets()
   * \brief Search the string variable wrapped as glvar_t from the pool by a
   * string type key and return the result. \param key The key to indicate the
   * global variable. \param val The string type container to receive the
   * result. \return True if found, otherwise false.
   */
  bool gets(const string key, string &val);
  /**
   * \fn put()
   * \brief Update the glvar_t variable in the pool located by a string type key
   * and return the result. If the variable doesn't exist, then create a new one
   * and associate it with the given key. In either situation, the global
   * variable will be updated according to the given val. \param key The key to
   * indicate the global variable. \param val The value of glvar_t type
   * variable. \return True if the variable already exists, otherwise false.
   */
  bool put(const string key, const glvar_t val);
  /**
   * \fn putb()
   * \brief Update the bool variable wrapped as glvar_t in the pool located by a
   * string type key and return the result. If the variable doesn't exist, then
   * create a new one and associate it with the given key. In either situation,
   * the global variable will be updated according to the given val. \param key
   * The key to indicate the global variable. \param val The value of bool type
   * variable. \return True if the variable already exists, otherwise false.
   */
  bool putb(const string key, const bool val);
  /**
   * \fn puti()
   * \brief Update the int variable wrapped as glvar_t in the pool located by a
   * string type key and return the result. If the variable doesn't exist, then
   * create a new one and associate it with the given key. In either situation,
   * the global variable will be updated according to the given val. \param key
   * The key to indicate the global variable. \param val The value of int type
   * variable. \return True if the variable already exists, otherwise false.
   */
  bool puti(const string key, const int val);
  /**
   * \fn putf()
   * \brief Update the float variable wrapped as glvar_t in the pool located by
   * a string type key and return the result. If the variable doesn't exist,
   * then create a new one and associate it with the given key. In either
   * situation, the global variable will be updated according to the given val.
   * \param key The key to indicate the global variable.
   * \param val The value of float type variable.
   * \return True if the variable already exists, otherwise false.
   */
  bool putf(const string key, const float val);
  /**
   * \fn puts()
   * \brief Update the string variable wrapped as glvar_t in the pool located by
   * a string type key and return the result. If the variable doesn't exist,
   * then create a new one and associate it with the given key. In either
   * situation, the global variable will be updated according to the given val.
   * \param key The key to indicate the global variable.
   * \param val The value of string type variable.
   * \return True if the variable already exists, otherwise false.
   */
  bool puts(const string key, const string val);

  /**
   * \fn load_vars()
   * \brief Load variables from an external json configuration file.
   *
   * Json example:
   * [
   *   {
   *     "path": "/path1",
   *     "variables": [
   *       {
   *         "name": "var1",
   *         "type": "boolean",
   *         "value": false
   *       },
   *       {
   *         "name": "var2",
   *         "type": "integer",
   *         "value": 2
   *       }
   *     ]
   *   }
   * ]
   *
   * \param filename_ The name of the configuration file.
   * \return True if it's successful, otherwise false.
   */
  bool load_vars(const string filename_);

  /**
   * \fn store_vars()
   * \brief Store variables from an external json configuration file.
   *
   * \param filename_ The name of the configuration file.
   * \return True if it's successful, otherwise false.
   */
  bool store_vars(const string filename_);
  /**
   * \fn select_and_store_vars()
   * \brief Select and store variables from an external json configuration file.
   * This function has a white filter. It only stores the variables that are
   * listed in the filer list.
   *
   * \param filename_ The name of the configuration file.
   * \param filter_ The white filter list.
   * \return True if it's successful, otherwise false.
   */
  bool select_and_store_vars_white(const string filename_,
                                   std::set<string> filter_);
  /**
   * \fn select_and_store_vars()
   * \brief Select and store variables from an external json configuration file.
   * This function has a black filter. It stores all the variables except those
   * listed in the filer list.
   *
   * \param filename_ The name of the configuration file.
   * \param filter_ The black filter list.
   * \return True if it's successful, otherwise false.
   */
  bool select_and_store_vars_black(const string filename_,
                                   std::set<string> filter_);
  /**
   * \fn clear()
   * \brief Remove all variables.
   */
  void clear();

private:
  static map<string, glvar_t> _pool;
};

} // namespace util
} // namespace vesyla

#endif // __VS_UTIL_GLOBAL_VAR_HPP__
