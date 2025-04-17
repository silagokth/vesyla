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

#ifndef __VESYLA_UTIL_INTERVAL_HPP__
#define __VESYLA_UTIL_INTERVAL_HPP__

#include "easylogpp/easylogging++.h"
#include <limits.h>
#include <string>

using namespace std;

namespace vesyla
{
namespace util
{
class Interval
{
public:
	int lb;
	int hb;

public:
	Interval(int lb_ = 0, int hb_ = 0);
	~Interval();
	string to_str() const;
	bool empty() const;
	bool full() const;
	Interval operator+(const Interval &b_);
	Interval operator+(int b_);
	Interval operator-(const Interval &b_);
	Interval operator-(int b_);
	Interval operator&(const Interval &b_);
	Interval operator&(int b_);
	bool operator==(const Interval &b_);
	bool operator!=(const Interval &b_);
};
} // namespace util
} // namespace vesyla

#endif // __VESYLA_UTIL_OBJECT_HPP__