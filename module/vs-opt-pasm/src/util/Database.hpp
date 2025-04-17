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

#ifndef __VESYLA_UTIL_DATABASE_HPP__
#define __VESYLA_UTIL_DATABASE_HPP__

#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <cfloat>
#include <math.h>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <unordered_map>
#include <string>
#include "easylogpp/easylogging++.h"
#include "Object.hpp"
#include "SysPath.hpp"

using namespace std;

namespace vesyla
{
namespace util
{
class Database
{
private:
	std::unordered_map<string, Object *> _data;
	boost::property_tree::ptree _p;

public:
	Database();
	Database(string filename_);
	bool add(Object *obj_);
	bool remove(string name_);
	Object *get_object(string name_);
	void snapshot(string snapshot_name_);
	boost::property_tree::ptree get_object_ptree(string name_);
};
} // namespace util
} // namespace vesyla

#endif // __VESYLA_UTIL_DATABASE_HPP__