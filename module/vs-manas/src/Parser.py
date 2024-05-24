import DataBase_pb2 as db
import os
import sys
import re
import json
from google.protobuf.json_format import Parse
from lark import Lark, Tree, Token
from uuid import uuid4
import logging

def get_random_id():
    return "__"+str(uuid4())[:8]+"__"


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

        with open(filename, 'r') as f:
            prog = self.parse_asm(f.read())
        
        current_label = None
        for content in prog.contents:
            record = prog.records[content]
            if record.kind == db.ASMRecord.CELL:
                x = record.parameters["x"]
                y = record.parameters["y"]
                current_label = f"{x}_{y}"
            elif record.kind == db.ASMRecord.INSTR:
                if not record.id.startswith("__"):
                    label_pc_map[record.id] = len(data.pkg.instruction_lists[current_label].instructions)
                if current_label is None:
                    logging.error("Instruction without cell")
                    return False

                # Find the instruction template in the database by command, comparison is case insensitive.
                template = None
                for instr in data.isa.instruction_templates:
                    if instr.name == record.name:
                        template = instr
                        break
                if template == None:
                    print("Error: Instruction template not found: ", record.name)
                    return False
                # create a new instruction
                instr = db.Instruction()
                # set the instruction name
                instr.name = record.name
                # use default value for each segment
                for segment in template.segment_templates:
                    vm = db.StrIntMap()
                    vm.key = segment.name
                    vm.val = segment.default_val
                    instr.value_map.append(vm)
                
                for key in record.parameters:
                    val = record.parameters[key]
                    arg_found = False
                    for segment in template.segment_templates:
                        if segment.name == key:
                            arg_found = True
                            verbo_map = segment.verbo_map
                            break
                    if not arg_found:
                        print("Error: Invalid argument name:", key)
                        return False
                    for vm in instr.value_map:
                        if vm.key == key:
                            factor = 1
                            if val[0] == "-":
                                factor = -1
                                val = val[1:]
                            if val[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                                vm.val = factor * self.conv_num(val)
                            else:
                                found_verbo_str = False
                                if not len(verbo_map) == 0:
                                    for e in verbo_map:
                                        if e.val == val:
                                            vm.val = e.key
                                            found_verbo_str = True
                                            break
                                if not found_verbo_str:
                                    vm.val = 0
                                    cell_index = -1
                                    for i in range(len(data.pkg.instruction_lists)):
                                        if data.pkg.instruction_lists[i].label == current_label:
                                            cell_index = i
                                            break
                                    if cell_index == -1:
                                        cell_index = len(data.pkg.instruction_lists)
                                    i = cell_index
                                    j = len(data.pkg.instruction_lists[cell_index].instructions)
                                    translation_table["{}_{}_{}".format(i, j, vm.key)] = val
                            break
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
                data.pkg.instruction_lists[cell_index].instructions.append(instr)

        # convert labels to pc
        for i in range(len(data.pkg.instruction_lists)):
            for j in range(len(data.pkg.instruction_lists[i].instructions)):
                instr = data.pkg.instruction_lists[i].instructions[j]
                for vm in instr.value_map:
                    if "{}_{}_{}".format(i, j, vm.key) in translation_table:
                        vm.val = label_pc_map[translation_table["{}_{}_{}".format(
                            i, j, vm.key)]] - j

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
    
    def analyze_asm(self, node, prog=db.ASMProg()):
        if isinstance(node, Tree):
            if node.data == "start":
                prog = db.ASMProg()
                for child in node.children:
                    prog.contents.append(self.analyze_asm(child, prog).id)
                return prog
            elif node.data == "cell":
                cell = db.ASMRecord()
                cell.id = get_random_id()
                cell.kind = db.ASMRecord.CELL
                identifier_counter = 0
                for child in node.children:
                    if isinstance(child, Tree):
                        if child.data == "parameter":
                            if len(child.children) == 2:
                                ch0 = child.children[0]
                                ch1 = child.children[1]
                                if isinstance(ch0, Token) and isinstance(ch1, Token) and ch0.type == "IDENTIFIER" and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER"):
                                    cell.parameters[ch0.value] = ch1.value
                            else:
                                logging.error("Invalid parameter: " + str(child))
                                sys.exit(-1)
                    else:
                        cell.name = child
                prog.records[cell.id].CopyFrom(cell)
                return cell
            elif node.data == "instruction":
                instruction = db.ASMRecord()
                instruction.id = get_random_id()
                instruction.kind = db.ASMRecord.INSTR
                identifier_counter = 0
                for child in node.children:
                    if isinstance(child, Tree):
                        if child.data == "parameter":
                            if len(child.children) == 2:
                                ch0 = child.children[0]
                                ch1 = child.children[1]
                                if isinstance(ch0, Token) and isinstance(ch1, Token) and ch0.type == "IDENTIFIER" and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER"):
                                    instruction.parameters[ch0.value] = ch1.value
                            else:
                                logging.error("Invalid parameter: " + str(child))
                                sys.exit(-1)
                    else:
                        if identifier_counter == 0:
                            instruction.name = child
                            identifier_counter += 1
                        elif identifier_counter == 1:
                            instruction.id = child
                            identifier_counter += 1
                        else:
                            logging.error("Invalid instruction: " + str(node))
                            sys.exit(-1)
                
                prog.records[instruction.id].CopyFrom(instruction)
                return instruction

    def parse_asm(self, text_asm):
        asm_grammar = """
    start: (cell | instruction)*

    cell: "cell" ("<" IDENTIFIER ">")? "(" parameter ("," parameter)* ")"
    instruction: IDENTIFIER ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")")?
    parameter: IDENTIFIER "=" (IDENTIFIER | NUMBER)

    IDENTIFIER: /[_a-zA-Z][_a-zA-z0-9]*/
    NUMBER: /[+-]?(0[xdob])?[0-9\.]+/

    %import common.WS
    %ignore WS
    COMMENTS: /#.*/
    %ignore COMMENTS
    """
        parser = Lark(asm_grammar)
        program = parser.parse(text_asm)
        return self.analyze_asm(program)
