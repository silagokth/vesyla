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

#include "Interval.hpp"

namespace vesyla
{
namespace util
{
Interval::Interval(int lb_, int hb_)
{
	lb = lb_;
	hb = hb_;
}
Interval::~Interval()
{
}
string Interval::to_str() const
{
	string str = "[" + to_string(lb) + ", " + to_string(hb) + "]";
	return str;
}
bool Interval::empty() const
{
	if (lb > hb)
	{
		return true;
	}
	if (lb == hb && (lb == INT_MIN || hb == INT_MAX))
	{
		return true;
	}
	return false;
}
bool Interval::full() const
{
	if (lb == INT_MIN && hb == INT_MAX)
	{
		return true;
	}
	return false;
}
Interval Interval::operator+(const Interval &b_)
{
	long llb0 = lb;
	long lhb0 = hb;
	long llb1 = b_.lb;
	long lhb1 = b_.hb;
	long llb2;
	long lhb2;
	if ((llb0 == INT_MIN) || (llb1 == INT_MIN))
	{
		llb2 = INT_MIN;
	}
	else
	{
		llb2 = llb0 + llb1;
	}

	if ((lhb0 == INT_MAX) || (lhb1 == INT_MAX))
	{
		lhb2 = INT_MAX;
	}
	else
	{
		lhb2 = lhb0 + lhb1;
	}

	if (llb2 <= INT_MIN)
	{
		llb2 = INT_MIN;
	}
	if (lhb2 >= INT_MAX)
	{
		lhb2 = INT_MAX;
	}
	Interval r((int)(llb2), (int)(lhb2));
	return r;
}
Interval Interval::operator+(int b_)
{
	Interval i(b_, b_);
	return operator+(i);
}
Interval Interval::operator-(const Interval &b_)
{
	Interval i(-b_.hb, -b_.lb);
	return operator+(i);
}
Interval Interval::operator-(int b_)
{
	Interval i(-b_, -b_);
	return operator+(i);
}
Interval Interval::operator&(const Interval &b_)
{
	Interval i(max(lb, b_.lb), min(hb, b_.hb));
	return i;
}
Interval Interval::operator&(int b_)
{
	Interval i(b_, b_);
	return operator&(i);
}
bool Interval::operator==(const Interval &b_)
{
	return lb == b_.lb && hb == b_.hb;
}
bool Interval::operator!=(const Interval &b_)
{
	return !(*this == b_);
}
} // namespace util
} // namespace vesyla