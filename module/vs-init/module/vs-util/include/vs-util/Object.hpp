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

#ifndef __VESYLA_UTIL_OBJECT_HPP__
#define __VESYLA_UTIL_OBJECT_HPP__

#include <string>
#include <boost/property_tree/ptree.hpp>

using namespace std;

namespace vesyla
{
namespace util
{
class Object
{
public:
	string name;
	Object();
	Object(boost::property_tree::ptree p_);
	virtual boost::property_tree::ptree serialize();
	//virtual void *translate(int id_) = 0;
};
} // namespace util
} // namespace vesyla

#endif // __VESYLA_UTIL_OBJECT_HPP__