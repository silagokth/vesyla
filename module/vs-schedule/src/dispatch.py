import os
import sys
import regex as re
import logging
import pyparsing as pp
import sync
import schedule
import generate

class EpochRegion:
    def __init__(self, name, content):
        self.name = name
        self.content = content
    def __str__(self):
        return "EpochBlock: " + str(self.name)

class LoopRegion:
    def __init__(self, name, count, content):
        self.name = name
        self.count = count
        self.content = content
    def __str__(self):
        return "LoopRegion " + str(self.name) + " " + str(self.count) + ": [ " + ", ".join([str(x) for x in self.content]) +" ]"

class ProgramRegion:
    def __init__(self, content):
        self.content = content
    def __str__(self):
        return "ProgramRegion: [ "+", ".join([str(x) for x in self.content])+" ]"

class CstrRegion:
    def __init__(self, name, content):
        self.name = name
        self.content = content
    def __str__(self):
        return "CstrBlock: " + str(self.name)

def parse_pasm(file_path):
    epoch_region = pp.Group(pp.Forward("epoch")+pp.Word(pp.alphanums)+"{"+pp.SkipTo("}")+"}")
    loop_region = pp.Group(pp.Forward("loop")+pp.Word(pp.alphanums)+pp.Word(pp.nums)+"{"+pp.Group(pp.OneOrMore(epoch_region))+"}")
    program = pp.Group(pp.ZeroOrMore(epoch_region | loop_region))
    program.ignore(pp.pythonStyleComment)
    
    with open(file_path, "r") as file:
        result = program.parseString(file.read())

    return result

def parse_cstr(file_path):
    epoch_region = pp.Group(pp.Forward("epoch")+pp.Word(pp.alphanums)+"{"+pp.SkipTo("}")+"}")
    program = pp.Group(pp.ZeroOrMore(epoch_region))
    program.ignore(pp.pythonStyleComment)

    with open(file_path, "r") as file:
        result = program.parseString(file.read())

    return result

def analyze_block(block):
    if block[0] == "epoch":
        return EpochRegion(block[1], block[3])
    elif block[0] == "loop":
        block_list = []
        for sub_block in block[4]:
            block_list.append(analyze_block(sub_block))
        return LoopRegion(block[1], block[2], block_list)
    else:
        logging.error("Unknown block type: %s", block[0])
        sys.exit(1)

def construct_ast(parsed_result):
    ast = ProgramRegion([])
    for block in parsed_result[0]:
        ast.content.append(analyze_block(block))
    return ast

def construct_cstr_list(parsed_result):
    cstr_list = []
    for block in parsed_result[0]:
        cstr_list.append(CstrRegion(block[1], block[3]))
    return cstr_list

def find_instruction_count(asm_txt):
    instr_count = {}
    curr_cell = None
    for line in asm_txt.split("\n"):
        # remove comments
        line = re.sub(r'#.*$', '', line)
        # remove leading and trailing whitespaces
        line = line.strip()
        # skip empty lines
        if not line:
            continue
        pattern = re.compile(r'cell\s+(\d+_\d+)')
        match = pattern.search(line)
        if match:
            cell = match.group(1)
            if cell not in instr_count:
                instr_count[cell] = 0
            curr_cell = cell
        else:
            instr_count[curr_cell] += 1
    return instr_count

def shift_loop_level(asm, shift_level):
    new_asm = ""
    for line in asm.split("\n"):
        # remove comments
        line = re.sub(r'#.*$', '', line)
        # remove leading and trailing whitespaces
        line = line.strip()
        # skip empty lines
        if not line:
            continue
        pattern = re.compile(r'looph\s+(.*)$')
        match = pattern.search(line)
        if match:
            parameters = match.group(1).split(",")
            for i in range(len(parameters)):
                parameter = parameters[i]
                key, value = parameter.split("=")
                if key.strip() == "id":
                    loop_id = int(value) + shift_level
                    parameters[i] = "id=%d" % loop_id
                    break
            new_asm += "looph " + ",".join(parameters) + "\n"
            continue
        pattern = re.compile(r'loopt\s+(.*)$')
        match = pattern.search(line)
        if match:
            parameters = match.group(1).split(",")
            for i in range(len(parameters)):
                parameter = parameters[i]
                key, value = parameter.split("=")
                if key.strip() == "id":
                    loop_id = int(value) + shift_level
                    parameters[i] = "id=%d" % loop_id
                    break
            new_asm += "loopt " + ",".join(parameters) + "\n"
            continue
        new_asm += line + "\n"
    return new_asm

def schedule_block(block, cstr_list, cells, loop_id_counter, output_dir):
    if isinstance(block, EpochRegion):
        # create a working directory, name it after the resource block
        os.makedirs(os.path.join(output_dir, block.name), exist_ok=True)
        # generate pasm file
        with open(os.path.join(output_dir, block.name, "0.pasm"), "w") as file:
            file.write(block.content)
        # generate cstr file
        with open(os.path.join(output_dir, block.name, "0.cstr"), "w") as file:
            for cstr_block in cstr_list:
                if cstr_block.name == block.name:
                    file.write(cstr_block.content)
                    break
        # generate model
        generate.generate(os.path.join(output_dir, block.name, "0.pasm"), os.path.join(output_dir, block.name, "0.cstr"), os.path.join(output_dir, block.name))
        schedule.schedule(os.path.join(output_dir, block.name, "model.txt"), os.path.join(output_dir, block.name))
        sync.sync_resource(os.path.join(output_dir, block.name, "0.pasm"), os.path.join(os.path.join(output_dir, block.name), "timing_table.json"), cells, os.path.join(output_dir, block.name))
        with open(os.path.join(output_dir, block.name, "instr_lists.txt"), "r") as file:
            return file.read()
    elif isinstance(block, LoopRegion):
        txt = ""
        for cell in cells:
            txt += "cell %s\n" % cell
            txt += "looph id=%d, iter=%s\n" % (loop_id_counter, block.count)

        asm_txt = ""
        for sub_block in block.content:
            asm_txt += schedule_block(sub_block, cstr_list, cells, loop_id_counter+1, os.path.join(output_dir, block.name))
        # find how many instructions are in the sub_block for each cell
        instr_count = find_instruction_count(asm_txt)
        for cell in cells:
            if cell not in instr_count:
                instr_count[cell] = 0
        asm_txt = shift_loop_level(asm_txt, 1)
        txt += asm_txt

        for cell in cells:
            txt += "cell %s\n" % cell
            txt += "loopt id=%d, pc=%d\n" % (loop_id_counter, -(instr_count[cell]))
        return txt
    elif isinstance(block, ProgramRegion):
        txt = ""
        for sub_block in block.content:
            txt += schedule_block(sub_block, cstr_list, cells, loop_id_counter, output_dir)
        return txt
    else:
        logging.error("Unknown block type: %s", block)
        sys.exit(1)

def find_all_cells(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    cells = set()
    for line in lines:
        # remove comments
        line = re.sub(r'#.*$', '', line)
        # remove leading and trailing whitespaces
        line = line.strip()
        # skip empty lines
        if not line:
            continue
        pattern = re.compile(r'cell\s+(\d+_\d+)')
        match = pattern.search(line)
        if match:
            cells.add(match.group(1))
    return cells


def dispatch(file_pasm, file_cstr, output_dir):
    cells = find_all_cells(file_pasm)
    program_pasm = parse_pasm(file_pasm)
    program_cstr = parse_cstr(file_cstr)
    ast = construct_ast(program_pasm)
    cstr_list = construct_cstr_list(program_cstr)
    txt = schedule_block(ast, cstr_list, cells, 0, output_dir)
    with open(os.path.join(output_dir, "instr_lists.asm"), "w") as file:
        file.write(txt)
