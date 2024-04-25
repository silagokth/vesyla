import os
import sys
import re
import logging

def parse_proto_asm(proto_asm_file):
    op_table = {}
    with open(proto_asm_file, 'r') as f:
        current_op = None
        for line in f:
            # remove all comments starting with #
            line = re.sub(r'#.*$', '', line)
            line = line.lower() # case insensitive
            line = line.strip()
            if line == '':
                continue
            pattern = re.compile(r'^cell\s+(\w+)$')
            match = pattern.match(line)
            if match is not None:
                continue
            pattern = re.compile(r'^operation\s+(\w+)\s+(.+)$')
            match = pattern.match(line)
            if match is not None:
                op_name = match.group(1)
                fields = match.group(2).split(',')
                slot =0
                port = 0
                for i in range(len(fields)):
                    fields[i] = re.sub(r'\s+', '', fields[i])
                    pattern = re.compile(r'(\w+)\=(.+)')
                    match = pattern.match(fields[i])
                    if match is None:
                        logging.error('Invalid field format: %s' % fields[i])
                        sys.exit(1)
                    field_name = match.group(1)
                    field_value = match.group(2)
                    if field_name == 'slot':
                        slot = int(field_value)
                    elif field_name == 'port':
                        port = int(field_value)
                    else:
                        logging.error('Invalid field name: %s' % field_name)
                        sys.exit(1)

                op = {'name': op_name, 'slot': slot, 'port': port, 'instr_list': []}
                op_table[op['name']] = op
                current_op = op
                continue
            pattern = re.compile(r'^(.+)$')
            match = pattern.match(line)
            if match is not None:
                instr = match.group(1)
                current_op['instr_list'].append(instr)
                continue
            
    return op_table

def extract_op_expr(op):
    T={}
    R={}
    expr = op['name']+"_e0"
    for instr in op['instr_list']:
        instr = instr.lower() # case insensitive
        instr = instr.strip()
        pattern = re.compile(r'rep\s+(.*)$')
        match = pattern.match(instr)
        if match is not None:
            level = 0
            iter = "1"
            delay = "0"
            fields = match.group(1).split(',')
            for field in fields:
                field = re.sub(r'\s+', '', field)
                pattern = re.compile(r'(\w+)\=(.+)')
                match = pattern.match(field)
                if match is None:
                    logging.error('Invalid field format: %s' % field)
                    sys.exit(1)
                field_name = match.group(1)
                field_value = match.group(2)
                if field_name == 'iter':
                    iter = field_value
                    # check if iter is a number
                    if not iter.isdigit():
                        logging.error('Invalid iter value: %s. Iteration must be a number!' % iter)
                        sys.exit(1)
                elif field_name == 'delay':
                    delay = field_value
                elif field_name == 'level':
                    level = int(field_value)
            R[level] = [iter, delay]
            continue
        pattern = re.compile(r'fsm\s+(.*)$')
        match = pattern.match(instr)
        if match is not None:
            T['delay_0'] = "0"
            T['delay_1'] = "0"
            T['delay_2'] = "0"
            fields = match.group(1).split(',')
            for field in fields:
                field = re.sub(r'\s+', '', field)
                pattern = re.compile(r'(\w+)\=(.+)')
                match = pattern.match(field)
                if match is None:
                    logging.error('Invalid field format: %s' % field)
                    sys.exit(1)
                field_name = match.group(1)
                field_value = match.group(2)
                if field_name == 'delay_0':
                    T['delay_0'] = field_value
                elif field_name == 't1':
                    T['delay_1'] = field_value
                elif field_name == 't2':
                    T['delay_2'] = field_value
            continue

    event_counter = 1
    for i in range(3):
        field_name = "delay_"+ str(i)
        if field_name in T and T[field_name] != "0":
            expr = "T<"+T[field_name]+">("+expr+", "+op['name']+"_e"+str(event_counter)+")"
            event_counter += 1
        else:
            break
    for i in range(8):
        if i in R:
            expr = "R<"+R[i][0]+","+R[i][1]+">("+expr+")"
        else:
            break
    return expr

def parse_constraint(constraint_file):
    constraint_list = []
    with open(constraint_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            if line[0] == "#":
                continue
            constraint_list.append(line)
    return constraint_list

def extract_anchor_symbols(expr, op_repeat_level, symbol_table, anchor_table):
    symbols = []
    pattern = re.compile(r'([a-zA-Z_$][\w]*)(\.e[0-9]+)?([\[\d\]]*)')
    matches = pattern.findall(expr)
    for match in matches:
        if match not in symbols:
            print(match)
            op_name = match[0]
            if op_name not in op_repeat_level:
                logging.error('Invalid operation name: %s' % op_name)
                sys.exit(1)
            event_name = match[1]
            event_name_remove_dot = event_name
            if event_name != "":
                event_name_remove_dot = event_name[1:] # remove the dot
            index_list = match[2]
            index_list_split = []
            if index_list != "":
                index_list_split = index_list[1:-1].split('][')
            repeat_level = op_repeat_level[op_name]
            full_match = op_name+event_name+index_list
            identifier_expr = translate_symbol_to_identifier(full_match, op_name, event_name_remove_dot, index_list_split, repeat_level)
            anchor_expr = translate_symbol_to_anchor(full_match, op_name, event_name_remove_dot, index_list_split, repeat_level)
            if identifier_expr not in symbol_table:
                symbol_table[full_match] = identifier_expr
                if identifier_expr != anchor_expr:
                    anchor_table[identifier_expr] = anchor_expr
            symbols.append(full_match)
    return symbols

def translate_symbol_to_identifier(full_expr, op_name, event_name, index_list, repeat_level):
    if event_name == "":
        event_name = "e0"
    if index_list == None:
        index_list = []
    expr = op_name +"_"+ event_name

    if len(index_list) > repeat_level:
        logging.error('%s: Index list slice out of range! %d slice requested, but the repetition level is only %d ' % (full_expr, len(index_list), repeat_level))
        sys.exit(1)

    for i in range(repeat_level):
        if i < len(index_list):
            expr = expr + "_" + index_list[i]
        else:
            expr = expr + "_0"
    return expr

def translate_symbol_to_anchor(full_expr, op_name, event_name, index_list, repeat_level):
    if event_name == "":
        event_name = "e0"
    if index_list == None:
        index_list = []
    expr = op_name +"_"+ event_name

    if len(index_list) > repeat_level:
        logging.error('%s: Index list slice out of range! %d slice requested, but the repetition level is only %d ' % (full_expr, len(index_list), repeat_level))
        sys.exit(1)

    for i in range(repeat_level):
        if i < len(index_list):
            expr = expr + "[" + index_list[i] + "]"
        else:
            expr = expr + "[0]"
    return expr

def find_repeat_level(op_expr_table):
    repeat_level = {}
    for op_name in op_expr_table:
        expr = op_expr_table[op_name]
        print(expr)
        pattern = re.compile(r'R<\d+,[^>]+>')
        matches = pattern.findall(expr)
        level = len(matches)
        repeat_level[op_name] = level
    return repeat_level

def generate(proto_asm_file, constraint_file, output_dir):
    op_table = parse_proto_asm(proto_asm_file)
    op_expr_table = {}
    for op_name in op_table:
        op = op_table[op_name]
        expr = extract_op_expr(op)
        op_expr_table[op_name] = expr
    op_repeat_level = find_repeat_level(op_expr_table)
    constraint_list = parse_constraint(constraint_file)
    symbol_table = {}
    anchor_table = {}
    for i in range(len(constraint_list)):
        constraint = constraint_list[i]
        symbol_list = extract_anchor_symbols(constraint, op_repeat_level, symbol_table, anchor_table)
        for symbol in symbol_list:
            constraint = constraint.replace(symbol, symbol_table[symbol])
        constraint_list[i] = constraint
    
    with open(os.path.join(output_dir, "model.txt"), 'w') as f:
        for op_name in op_expr_table:
            f.write("operation "+op_name+" "+op_expr_table[op_name]+"\n")
        for anchor in anchor_table:
            f.write("anchor "+anchor+" "+anchor_table[anchor]+"\n")
        for constraint in constraint_list:
            f.write("constraint "+constraint+"\n")
    
    logging.info("Model file is generated at %s" % os.path.join(output_dir, "model.txt"))

    
