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

#include "Database.hpp"

namespace pt = boost::property_tree;

namespace vesyla
{
namespace util
{
Database::Database()
{
}
Database::Database(string filename_)
{
	;
}
bool Database::add(Object *obj_)
{
	if (_data.find(obj_->name) != _data.end())
	{
		return false;
	}
	_data[obj_->name] = obj_;
	return true;
}

bool Database::remove(string name_)
{
	if (_data.find(name_) != _data.end())
	{
		_data.erase(name_);
		return true;
	}
	return false;
}

Object *Database::get_object(string name_)
{
	if (_data.find(name_) != _data.end())
	{
		return _data[name_];
	}
	return NULL;
}

void Database::snapshot(string snapshot_name_)
{
	LOG(INFO) << "Taking snapshot : " << snapshot_name_;

	vesyla::util::GlobalVar glv;
	string path;
	CHECK(glv.gets("$OUTPUT_DIR", path));
	path = path + "util/";
	mkdir(path.c_str(), 0755);
	path = path + "snapshots/";
	mkdir(path.c_str(), 0755);
	path = path + snapshot_name_ + "/";
	mkdir(path.c_str(), 0755);

	for (auto &o : _data)
	{
		string filename = path + o.first + ".xml";
		pt::ptree pi;
		pt::ptree pd;
		pi.put("name", o.first);
		pi.add_child("data", o.second->serialize());
		pd.add_child("object", pi);
		pt::write_xml(filename, pd);
	}
}

} // namespace util
} // namespace vesyla