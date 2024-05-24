import DataBase_pb2 as db

import os
import sys
from google.protobuf.json_format import MessageToJson


class CodeGenerator:
    def __init__(self):
        pass

    def to_json(self, data: db.DataBase, filename: str):
        # write data to json file
        with open(filename, 'w') as f:
            f.write(MessageToJson(data))

    def to_bin(self, data: db.DataBase, filename: str):
        # write data to binary file
        with open(filename, 'wb') as f:
            f.write(data.SerializeToString())

    def find_instr_template(self, instr_name: str, isa: db.InstructionSet) -> db.InstrctionTemplate:
        for template in isa.instruction_templates:
            if template.name.lower() == instr_name.lower():
                return template
        return None

    def int2bin_signed(self, num: int, bits: int) -> str:
        # convert signed integer to binary string
        if num < 0:
            return bin(num + (1 << bits))[2:]
        else:
            return bin(num)[2:].zfill(bits)

    def int2bin_unsigned(self, num: int, bits: int) -> str:
        # convert unsigned integer to binary string
        return bin(num)[2:].zfill(bits)

    def assemble_instr_bin(self, instr: db.Instruction, isa: db.InstructionSet) -> str:
        # assemble instruction from template
        content = ""
        template = self.find_instr_template(instr.name.lower(), isa)
        # convert instr.operands to binary string and add to content, number of bits is isa.instr_operand_bitwidth .
        content += self.int2bin_unsigned(template.code,
                                         isa.instr_code_bitwidth)
        for segment in template.segment_templates:
            if segment.is_signed:
                value = None
                for vm in instr.value_map:
                    if vm.key == segment.name:
                        value = vm.val
                        break
                if value == None:
                    print("Error: Segment value not found: ", segment.name)
                    return None
                content += self.int2bin_signed(value, segment.bitwidth)
            else:
                value = None
                for vm in instr.value_map:
                    if vm.key == segment.name:
                        value = vm.val
                        break
                if value == None:
                    print("Error: Segment value not found: ", segment.name)
                    return None
                content += self.int2bin_unsigned(value, segment.bitwidth)
        # fill the rest of the bits with zeros if needed
        content += "0" * (isa.instr_bitwidth - len(content))
        return content

    def assemble_instr_text(self, instr: db.Instruction, isa: db.InstructionSet) -> str:
        # assemble instruction from template
        content = ""
        template = self.find_instr_template(instr.name, isa)
        # convert instruction segment into a single line of text
        # The format is: instr_name(segment1_name = segment1_value, segment2_name = segment2_value, ... )
        content += template.name + " ("
        for segment in template.segment_templates:
            value = None
            for vm in instr.value_map:
                if vm.key == segment.name:
                    value = vm.val
                    break
            if value == None:
                print("Error: Segment value not found: ", segment.name)
                return None
            # if this is the last segment, do not add comma otherwise add comma
            if segment == template.segment_templates[-1]:
                content += segment.name + "=" + \
                    str(value)
            else:
                content += segment.name + "=" + \
                    str(value) + ", "
        content += ")"
        return content

    def dump_instr_bin(self, data: db.DataBase, filename: str):
        # write only instructions to binary file
        with open(filename, 'w') as f:
            for instr_list in data.pkg.instruction_lists:
                f.write("cell " + instr_list.label + '\n')
                for instr in instr_list.instructions:
                    bin_content = self.assemble_instr_bin(instr, data.isa)
                    f.write(bin_content + '\n')

    def dump_instr_text(self, data: db.DataBase, filename: str):
        # write only instructions to text file
        with open(filename, 'w') as f:
            for instr_list in data.pkg.instruction_lists:
                coord = instr_list.label.split("_")
                x = int(coord[0])
                y = int(coord[1])
                f.write("cell " + f"(x={x}, y={y})" + '\n')
                for instr in instr_list.instructions:
                    text_content = self.assemble_instr_text(instr, data.isa)
                    f.write(text_content + '\n')

    def run(self, data: db.DataBase, dir: str):
        self.dump_instr_bin(data, os.path.join(dir, "instr.bin"))
        self.dump_instr_text(data, os.path.join(dir, "instr.txt"))
        self.to_json(data, os.path.join(dir, "data.json"))
        self.to_bin(data, os.path.join(dir, "data.bin"))
