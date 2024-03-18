// Copyright (C) 2021 Yu Yang
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

#ifndef __VS_INIT_MAIN_HPP__
#define __VS_INIT_MAIN_HPP__

#include "vs-util/Common.hpp"
#include <ctime>
#include <iostream>
#include <libgen.h>
#include <linux/limits.h>
#include <string>
#include <unistd.h>

#include "VeConfig.hpp"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <filesystem>

int main(int argc, char **argv);
void lock(bool force, string path_output);
void initialize(string style, string path_output);

#endif // __VS_INIT_MAIN_HPP__
