import DataBase_pb2 as db
import os
import sys
import re
import json
from google.protobuf.json_format import Parse


class Parser:
    def __init__(self) -> None:
        pass

    def conv_bin(self, bin_str: str) -> int:
        '''Convert binary string to unsigned integer'''
        return int(bin_str, 2)

    def conv_hex(self, hex_str: str) -> int:
        '''Convert hex string to unsigned integer'''
        return int(hex_str, 16)

    def conv_oct(self, dec_str: str) -> int:
        '''Convert oct string to unsigned integer'''
        return int(dec_str, 8)

    def conv_num(self, num_str: str) -> int:
        if num_str.startswith('0x'):
            return self.conv_hex(num_str[2:])
        elif num_str.startswith('0b'):
            return self.conv_bin(num_str[2:])
        elif num_str.startswith('0o'):
            return self.conv_oct(num_str[2:])
        else:
            return int(num_str)
        return 0

    def load_instr(self, filename: str, data: db.DataBase) -> bool:
        label_pc_map = {}
        translation_table = {}

        # Preprocess the file, if a line end with ":" then connect it with the next line
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        for i in range(len(lines)):
            match = re.match(r"(.+):\s*$", lines[i])
            if match:
                lines[i] = match.group(1) + ": " + lines[i+1]
                lines[i+1] = ""

        lines = [line.strip() for line in lines if line.strip()
                 and not line.startswith('#')]
        # change all uppercase letters to lowercase
        lines = [line.lower() for line in lines]

        # parse each line using regex and keep the result. Each line has the following format:
        # <COMMAND>(<ARGUMENTS>)
        # if the format is wrong, return false
        current_label = ""
        for line in lines:
            match = re.match(r"\s*cell\s+(.*)", line)
            if match:
                current_label = match.group(1)
                continue

            match = re.match(r"\s*([^\s^:]+)\s*:\s*([^\s]+)\s+(.*)", line)
            if match:
                line_label = match.group(1)
                command = match.group(2)
                arguments = match.group(3)
            else:
                match = re.match(r"\s*([^\s]+)\s+(.*)", line)
                if match:
                    line_label = ""
                    command = match.group(1)
                    arguments = match.group(2)
                else:
                    match = re.match(r"\s*([^\s]+)\s*", line)
                    if match:
                        line_label = ""
                        command = match.group(1)
                        arguments = ""
                    else:
                        print("Error: Invalid line format: ", line)
                        return False

            if current_label == "":
                print("Error: Instruction without label")
                return False

            if line_label != "":
                cell_index = -1
                for i in range(len(data.pkg.instruction_lists)):
                    if data.pkg.instruction_lists[i].label == current_label:
                        cell_index = i
                        break
                if cell_index == -1:
                    label_pc_map[line_label] = len(
                        data.pkg.instruction_lists)
                else:
                    label_pc_map[line_label] = len(
                        data.pkg.instruction_lists[cell_index].instructions)

            # Find the instruction template in the database by command, comparison is case insensitive.
            template = None
            for instr in data.isa.instruction_templates:
                if instr.name.lower() == command:
                    template = instr
                    break
            if template == None:
                print("Error: Instruction template not found: ", command)
                return False
            # create a new instruction
            instr = db.Instruction()
            # set the instruction name
            instr.name = command
            # use default value for each segment
            for segment in template.segment_templates:
                vm = db.StrIntMap()
                vm.key = segment.name
                vm.val = segment.default_val
                instr.value_map.append(vm)

            # parse the ARGUMENTS field. The ARGUMENT field is a comma separated list of arguments with the following format:
            # <ARGUMENT_NAME> = <ARGUMENT_VALUE>
            if arguments != "":
                args = arguments.split(',')
                # set the instruction arguments
                for arg in args:
                    # use regex to remove all whitespaces like space and tab
                    arg = re.sub(r"\s+", "", arg)
                    arg_match = re.match(r"(.+)=(.+)", arg)
                    if arg_match:
                        # check if the argument name is in the template
                        arg_found = False
                        for segment in template.segment_templates:
                            if segment.name.lower() == arg_match.group(1):
                                arg_found = True
                                verbo_map = segment.verbo_map
                                break
                        if not arg_found:
                            print("Error: Invalid argument name:",
                                  arg_match.group(1))
                            return False

                        # set the value of the argument
                        for vm in instr.value_map:
                            if vm.key.lower() == arg_match.group(1):
                                value = arg_match.group(2)
                                factor = 1
                                if value[0] == "-":
                                    factor = -1
                                    value = value[1:]
                                if value[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                                    # number
                                    vm.val = factor * self.conv_num(value)
                                else:
                                    # verbo name or label
                                    found_verbo_str = False
                                    if not len(verbo_map) == 0:
                                        found_verbo_str = False
                                        for e in verbo_map:
                                            if e.val == value:
                                                vm.val = e.key
                                                found_verbo_str = True
                                                break
                                    if not found_verbo_str:
                                        # treat it as a label
                                        vm.val = 0
                                        cell_index = -1
                                        for i in range(len(data.pkg.instruction_lists)):
                                            if data.pkg.instruction_lists[i].label == current_label:
                                                cell_index = i
                                                break
                                        if cell_index == -1:
                                            cell_index = len(
                                                data.pkg.instruction_lists)
                                        i = cell_index
                                        j = len(
                                            data.pkg.instruction_lists[cell_index].instructions)
                                        translation_table["{}_{}_{}".format(
                                            i, j, vm.key.lower())] = value
                                break
                    else:
                        print("Error: Invalid argument format")
                        return False
                    # add the instruction to the current cell
                    # find the index of the current cell that matches the label, if label does not exist, create it.
            cell_index = -1
            for i in range(len(data.pkg.instruction_lists)):
                if data.pkg.instruction_lists[i].label == current_label:
                    cell_index = i
                    break
            if cell_index == -1:
                cell_index = len(data.pkg.instruction_lists)
                data.pkg.instruction_lists.append(
                    db.InstructionList())
                data.pkg.instruction_lists[cell_index].label = current_label
            data.pkg.instruction_lists[cell_index].instructions.append(
                instr)

        # convert labels to pc
        for i in range(len(data.pkg.instruction_lists)):
            for j in range(len(data.pkg.instruction_lists[i].instructions)):
                instr = data.pkg.instruction_lists[i].instructions[j]
                for vm in instr.value_map:
                    if "{}_{}_{}".format(i, j, vm.key.lower()) in translation_table:
                        vm.val = label_pc_map[translation_table["{}_{}_{}".format(
                            i, j, vm.key.lower())]] - j

        return True

    def load_isa(self, filename: str, data: db.DataBase) -> bool:
        # Load the instruction set file (json format) to data.isa (protobuf)
        text = ""
        with open(filename, 'r') as f:
            text = f.read()
        try:
            Parse(text, data.isa)
        except Exception as e:
            print("Error: Failed to parse instruction set file: ", e)
            return False
        return True
