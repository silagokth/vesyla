import sys
import logging
from lark import Lark, Tree, Token
from uuid import uuid4
import ds_pb2 as ds


def get_random_id():
    return "__" + str(uuid4())[:8] + "__"


# construct the AST for PASM
def analyze_pasm(node, prog=ds.PASMProg()):
    if isinstance(node, Tree):
        if node.data == "start":
            prog = ds.PASMProg()
            region = ds.PASMRecord()
            region.id = get_random_id()
            region.kind = ds.PASMRecord.START
            prog.start = region.id
            for child in node.children:
                region.contents.append(analyze_pasm(child, prog).id)
            prog.records[region.id].CopyFrom(region)
            return prog
        elif (
            node.data == "loop_region"
            or node.data == "cond_region"
            or node.data == "epoch_region"
            or node.data == "cell_region"
            or node.data == "cop_region"
            or node.data == "rop_region"
            or node.data == "raw_region"
        ):

            if node.data == "loop_region":
                region = ds.PASMRecord()
                region.id = get_random_id()
                region.kind = ds.PASMRecord.LOOP
            elif node.data == "cond_region":
                region = ds.PASMRecord()
                region.id = get_random_id()
                region.kind = ds.PASMRecord.COND
            elif node.data == "epoch_region":
                region = ds.PASMRecord()
                region.id = get_random_id()
                region.kind = ds.PASMRecord.EPOCH
            elif node.data == "cell_region":
                region = ds.PASMRecord()
                region.id = get_random_id()
                region.kind = ds.PASMRecord.CELL
            elif node.data == "cop_region":
                region = ds.PASMRecord()
                region.id = get_random_id()
                region.kind = ds.PASMRecord.COP
            elif node.data == "rop_region":
                region = ds.PASMRecord()
                region.id = get_random_id()
                region.kind = ds.PASMRecord.ROP
            elif node.data == "raw_region":
                region = ds.PASMRecord()
                region.id = get_random_id()
                region.kind = ds.PASMRecord.RAW

            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "parameter":
                        if len(child.children) == 2:
                            ch0 = child.children[0]
                            ch1 = child.children[1]
                            if (
                                isinstance(ch0, Token)
                                and isinstance(ch1, Token)
                                and ch0.type == "IDENTIFIER"
                                and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER")
                            ):
                                region.parameters[ch0.value] = ch1.value
                        else:
                            logging.error("Invalid parameter: " + str(child))
                            sys.exit(-1)

                    else:
                        child_record = analyze_pasm(child, prog)
                        region.contents.append(child_record.id)
                else:
                    region.id = child
            prog.records[region.id].CopyFrom(region)
            return region
        elif node.data == "instruction":
            instruction = ds.PASMRecord()
            instruction.id = get_random_id()
            instruction.kind = ds.PASMRecord.INSTR
            identifier_counter = 0
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "parameter":
                        if len(child.children) == 2:
                            ch0 = child.children[0]
                            ch1 = child.children[1]
                            if (
                                isinstance(ch0, Token)
                                and isinstance(ch1, Token)
                                and ch0.type == "IDENTIFIER"
                                and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER")
                            ):
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


# construct the AST for CSTR
def analyze_cstr(node, prog=ds.CSTRProg()):
    if isinstance(node, Tree):
        if node.data == "start":
            prog = ds.CSTRProg()
            region = ds.CSTRRecord()
            region.id = get_random_id()
            region.kind = ds.CSTRRecord.START
            prog.start = region.id
            for child in node.children:
                region.contents.append(analyze_cstr(child, prog).id)
            prog.records[region.id].CopyFrom(region)
            return prog
        elif node.data == "epoch_region":
            region = ds.CSTRRecord()
            region.id = get_random_id()
            region.kind = ds.CSTRRecord.EPOCH

            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "parameter":
                        if len(child.children) == 2:
                            ch0 = child.children[0]
                            ch1 = child.children[1]
                            if (
                                isinstance(ch0, Token)
                                and isinstance(ch1, Token)
                                and ch0.type == "IDENTIFIER"
                                and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER")
                            ):
                                region.parameters[ch0.value] = ch1.value
                        else:
                            logging.error("Invalid parameter: " + str(child))
                            sys.exit(-1)

                    else:
                        child_record = analyze_cstr(child, prog)
                        region.contents.append(child_record.id)
                else:
                    region.id = child
            prog.records[region.id].CopyFrom(region)
            return region
        elif node.data == "constraint":
            constraint = ds.CSTRRecord()
            constraint.id = get_random_id()
            constraint.kind = ds.CSTRRecord.CSTR
            for child in node.children:
                if isinstance(child, Token):
                    if child.type == "IDENTIFIER":
                        constraint.name = child.value
                    else:
                        constraint.constraint = child.value

            prog.records[constraint.id].CopyFrom(constraint)
            return constraint


def analyze_asm(node, prog=ds.ASMProg()):
    if isinstance(node, Tree):
        if node.data == "start":
            prog = ds.ASMProg()
            for child in node.children:
                prog.contents.append(analyze_asm(child, prog).id)
            return prog
        elif node.data == "cell":
            cell = ds.ASMRecord()
            cell.id = get_random_id()
            cell.kind = ds.ASMRecord.CELL
            identifier_counter = 0
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "parameter":
                        if len(child.children) == 2:
                            ch0 = child.children[0]
                            ch1 = child.children[1]
                            if (
                                isinstance(ch0, Token)
                                and isinstance(ch1, Token)
                                and ch0.type == "IDENTIFIER"
                                and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER")
                            ):
                                cell.parameters[ch0.value] = ch1.value
                        else:
                            logging.error("Invalid parameter: " + str(child))
                            sys.exit(-1)
                else:
                    cell.name = child
            prog.records[cell.id].CopyFrom(cell)
            return cell
        elif node.data == "instruction":
            instruction = ds.ASMRecord()
            instruction.id = get_random_id()
            instruction.kind = ds.ASMRecord.INSTR
            identifier_counter = 0
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "parameter":
                        if len(child.children) == 2:
                            ch0 = child.children[0]
                            ch1 = child.children[1]
                            if (
                                isinstance(ch0, Token)
                                and isinstance(ch1, Token)
                                and ch0.type == "IDENTIFIER"
                                and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER")
                            ):
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


def parse_pasm(text_pasm):
    grammar = """
start: (loop_region | cond_region | epoch_region)*

loop_region: "loop"   ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")") "{" (loop_region | cond_region | epoch_region)+ "}"
cond_region: "cond"   ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")") "{" (loop_region | cond_region | epoch_region)+ "}"
epoch_region: "epoch" ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")")? "{" cell_region+ "}"
cell_region: "cell"   ("<" IDENTIFIER ">")? "(" parameter ("," parameter)* ")" "{" (cop_region | rop_region | raw_region)+ "}"

cop_region: "cop" ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")")? "{" instruction+ "}"
rop_region: "rop" ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")")? "{" instruction+ "}"
raw_region: "raw" ("<" IDENTIFIER ">")? "{" instruction+ "}"

instruction: IDENTIFIER ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")")?
parameter: IDENTIFIER "=" (IDENTIFIER | NUMBER)

IDENTIFIER: /[_a-zA-Z][_a-zA-z0-9]*/
NUMBER: /[+-]?(0[xdob])?[0-9\.]+/

%import common.WS
%ignore WS
COMMENTS: /#.*/
%ignore COMMENTS
"""
    parser = Lark(grammar)  # Scannerless Earley is the default
    program = parser.parse(text_pasm)
    return analyze_pasm(program)


def parse_cstr(text_cstr):
    grammar = """
start: (epoch_region)*
epoch_region: "epoch" ("<" IDENTIFIER ">")? ("(" parameter ("," parameter)* ")")? "{" constraint* "}"
constraint: IDENTIFIER ANY
parameter: IDENTIFIER "=" (IDENTIFIER | NUMBER)
IDENTIFIER: /[_a-zA-Z][_a-zA-z0-9]*/
NUMBER: /[+-]?(0[xdob])?[0-9\.]+/
ANY: /.+/
%import common.WS
%ignore WS
COMMENTS: /#.*/
%ignore COMMENTS
"""
    parser = Lark(grammar)
    program = parser.parse(text_cstr)
    return analyze_cstr(program)


def parse_asm(text_asm):
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
    return analyze_asm(program)


def load_pasm_from_json(json_text):
    prog = ds.PASMProg()
    prog.ParseFromString(json_text)
    return prog


def load_cstr_from_json(json_text):
    prog = ds.CSTRProg()
    prog.ParseFromString(json_text)
    return prog


def load_asm_from_json(json_text):
    prog = ds.ASMProg()
    prog.ParseFromString(json_text)
    return prog


def analyze_asm_for_epoch(node, prog=ds.ASMProg()):
    if isinstance(node, Tree):
        if node.data == "start":
            prog = ds.ASMProg()
            for child in node.children:
                prog.contents.append(analyze_asm_for_epoch(child, prog).id)
            return prog
        elif node.data == "cell":
            cell = ds.ASMRecord()
            cell.id = get_random_id()
            cell.kind = ds.ASMRecord.CELL
            identifier_counter = 0
            for child in node.children:
                if isinstance(child, Token):
                    label = child.split("_")
                    x = label[0]
                    y = label[1]
                    cell.parameters["x"] = x
                    cell.parameters["y"] = y
            prog.records[cell.id].CopyFrom(cell)
            return cell
        elif node.data == "instruction":
            instruction = ds.ASMRecord()
            instruction.id = get_random_id()
            instruction.kind = ds.ASMRecord.INSTR
            identifier_counter = 0
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "parameter":
                        if len(child.children) == 2:
                            ch0 = child.children[0]
                            ch1 = child.children[1]
                            if (
                                isinstance(ch0, Token)
                                and isinstance(ch1, Token)
                                and ch0.type == "IDENTIFIER"
                                and (ch1.type == "IDENTIFIER" or ch1.type == "NUMBER")
                            ):
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


def text_to_asm_epoch(text_asm):
    asm_grammar = """
start: (cell | instruction)*

cell: "cell" CELL_LABEL
instruction: IDENTIFIER ( parameter ("," parameter)* )?
parameter: IDENTIFIER "=" (IDENTIFIER | NUMBER)

CELL_LABEL: /[0-9]+_[0-9]+/
IDENTIFIER: /[_a-zA-Z][_a-zA-z0-9]*/
NUMBER: /[+-]?(0[xdob])?[0-9\.]+/

%import common.WS
%ignore WS
COMMENTS: /#.*/
%ignore COMMENTS
"""
    parser = Lark(asm_grammar)
    program = parser.parse(text_asm)
    prog = analyze_asm_for_epoch(program)
    return prog
