// Copyright (C) 2022 Yu Yang
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

#ifndef __DRRA_HPP__
#define __DRRA_HPP__

#include "IO.hpp"
#include "Util.hpp"


void init(IO &input_buffer);
void model_l0(IO &input_buffer, IO &output_buffer);
void model_l1();

void store_data(std::string file_, IO buffer);
void load_data(std::string file_, IO &buffer);

void reset(IO &buffer);

void reset_all();
bool verify(std::string file0_, std::string file1_);

#endif // __DRRA_HPP__
