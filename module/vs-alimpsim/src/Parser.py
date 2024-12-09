import DataStructure_pb2 as ds
import os
import sys
import re
import json
from google.protobuf.json_format import Parse


class Parser:
    def __init__(self):
        pass

    def load_arch(self, arch: ds.ArchitectureDescription, filename: str) -> bool:
        '''Load json file into ArchitectureDescription protobuf object'''
        try:
            with open(filename, 'r') as f:
                json_str = f.read()
            Parse(json_str, arch)
            return True
        except Exception as e:
            print(e)
            return False

    def load_isa(self, isa: ds.InstructionSet, filename: str) -> bool:
        '''Load json file into InstructionSet protobuf object'''
        try:
            with open(filename, 'r') as f:
                json_str = f.read()
            Parse(json_str, isa)
            return True
        except Exception as e:
            print(e)
            return False

    def load_instr(self, arch: ds.ArchitectureDescription, pkg: ds.InstructionPackage, isa: ds.InstructionSet, filename: str) -> bool:
        '''Load text file into InstructionPackage protobuf object. Each line is either a coordinate tag or a single instruction'''
        lines = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        # filter all empty lines
        lines = [line for line in lines if line.strip() != ""]
        # change all uppercase letters to lowercase
        lines = [line.lower() for line in lines]

        current_label = ""
        for line in lines:
            # filter all comments starting with #
            line = re.sub(r"#.*", "", line)

            # match "cell <label>"
            match = re.match(r"\s*cell\s+(\d+)\s+(\d+)\s*", line)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                current_label = "{}_{}".format(x, y)
                continue

            # match instructions in N-bit binary form, only 0 and 1 are allowed. N is defined in the ISA.
            match = re.match(r"\s*([01]+)\s*", line)
            if match:
                # if no label is set, return false
                if current_label == "":
                    print("Error: Instruction without label")
                    return False
                # check if size = N
                if len(match.group(1)) != isa.instr_bitwidth:
                    print("Error: Instruction size not {} bits".format(
                        isa.instr_bitwidth))
                    print(line)
                    return False
                # IMPORTANT: reverse the binary string
                instr_str = match.group(1)[::-1]

                # create a new instruction
                instr = self.create_instr_from_bin(isa, instr_str)
                # add the instruction to the package
                # find the index of the current cell that matches the label, if label does not exist, create it.
                cell_index = -1
                for i in range(len(pkg.instruction_lists)):
                    if pkg.instruction_lists[i].label == current_label:
                        cell_index = i
                        break
                if cell_index == -1:
                    cell_index = len(pkg.instruction_lists)
                    pkg.instruction_lists.append(ds.InstructionList())
                    pkg.instruction_lists[cell_index].label = current_label
                pkg.instruction_lists[cell_index].instructions.append(instr)
                continue

        # add missing labels
        width = arch.fabric.width
        height = arch.fabric.height
        for r in range(height):
            for c in range(width):
                label = "{}_{}".format(r, c)
                found = False
                for instr_list in pkg.instruction_lists:
                    if instr_list.label == label:
                        found = True
                        break
                if not found:
                    instr_list = ds.InstructionList()
                    instr_list.label = label
                    pkg.instruction_lists.append(instr_list)

        # Now for every cell, check if the last instructions is a WIAT instruction with 0 cycles, if not add a WAIT instruction with 0 cycles.
        for instr_list in pkg.instruction_lists:
            if len(instr_list.instructions) == 0:
                # add a wait instruction
                instr = ds.Instruction()
                instr.name = "wait"
                instr.value_map.append(
                    ds.StrIntMap(key="mode", val=0))
                instr.value_map.append(
                    ds.StrIntMap(key="cycle", val=0))
                instr_list.instructions.append(instr)
            elif instr_list.instructions[-1].name == "wait":
                cycle = -1
                for vm in instr_list.instructions[-1].value_map:
                    if vm.key == "cycle":
                        cycle = vm.val
                        break
                if cycle != 0:
                    # add a wait instruction
                    instr = ds.Instruction()
                    instr.name = "wait"
                    instr.value_map.append(
                        ds.StrIntMap(key="mode", val=0))
                    instr.value_map.append(
                        ds.StrIntMap(key="cycle", val=0))
                    instr_list.instructions.append(instr)
            else:
                # add a wait instruction
                instr = ds.Instruction()
                instr.name = "wait"
                instr.value_map.append(
                    ds.StrIntMap(key="mode", val=0))
                instr.value_map.append(
                    ds.StrIntMap(key="cycle", val=0))
                instr_list.instructions.append(instr)

        return True

    def slice_instr_field(self, instr_str: str, hi, lo) -> str:
        '''Slice a field from a binary string'''
        return (instr_str[lo:hi])[::-1]

    def create_instr_from_bin(self, isa: ds.InstructionSet, instr_bin: str) -> ds.Instruction:
        '''Create an instruction from binary string'''
        # create a new instruction
        instr = ds.Instruction()
        # cut out the opcode from the binary string, the length of the opcode is defined in the instruction set. It start from the MSB side.
        opcode_str = instr_bin[isa.instr_bitwidth -
                               1:isa.instr_bitwidth-isa.instr_code_bitwidth-1:-1]
        # Convert opcode_str to opcode
        opcode = int(opcode_str, 2)

        # set the instruction name and segments
        for instr_template in isa.instruction_templates:
            if instr_template.code == opcode:
                instr.name = instr_template.name
                hi = isa.instr_bitwidth - isa.instr_code_bitwidth
                lo = hi
                for segment_template in instr_template.segment_templates:
                    lo = hi - segment_template.bitwidth
                    key = segment_template.name
                    val = int(self.slice_instr_field(instr_bin, hi, lo), 2)
                    if segment_template.is_signed:
                        if val >= 1 << (segment_template.bitwidth - 1):
                            val = val - (1 << segment_template.bitwidth)
                    instr.value_map.append(ds.StrIntMap(key=key, val=val))

                    hi = lo

        return instr

    def verify_db(self, db: ds.DataBase) -> bool:
        '''Verify the correctness of the database'''
        # check the number of instructions for each cell does not exceed the limit
        width = db.arch.fabric.width
        height = db.arch.fabric.height
        for r in range(height):
            for c in range(width):
                label = "{}_{}".format(r, c)
                # find out the controller for this cell
                controller = None
                cell_name = ""
                fabric = db.arch.fabric
                for cell in fabric.cell_lists:
                    for coord in cell.coordinates:
                        if r == coord.row and c == coord.col:
                            cell_name = cell.cell_name
                            break
                    if cell_name != "":
                        break

                if cell_name == "":
                    print("Error: Cell ({}, {}) not found".format(r, c))
                    return False

                for cell in db.arch.cells:
                    if cell.name == cell_name:
                        controller = cell.controller
                        break

                if controller == None:
                    print("Error: Controller for cell ({}, {}) not found".format(r, c))
                    return False

                # find out the iram_size from
                iram_size = controller.iram_size

                # find out the instruction list for this cell
                for il in db.pkg.instruction_lists:
                    if il.label == label:
                        if len(il.instructions) > iram_size:
                            print("Error: Too many instructions in cell ({}, {})".format(
                                r, c))
                            return False
                        break

        return True
