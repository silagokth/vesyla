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

#include <string>
using namespace std;

#ifndef __SWBInstruction_h__
#define __SWBInstruction_h__

#include "ComponentInfo.hpp"
#include "Instruction.hpp"

namespace BIR
{
class ComponentInfo;
// class Instruction;
class SWBInstruction;
} // namespace BIR

namespace BIR
{
class SWBInstruction : public BIR::Instruction
{
public:
	string variableName;
	BIR::ComponentInfo source;
	BIR::ComponentInfo destination;
	BIR::SWBInstruction *dependentTo;
	void load(pt::ptree p_);
	void load2(map<int, BIR::Instruction *> instr_list);
	pt::ptree dump();

	string to_bin();
	string to_str();
};
} // namespace BIR

#endif
