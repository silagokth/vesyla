import os
import sys
import re
import logging
import uuid
import operation
import itertools

def generate_uuid():
    return "__"+uuid.uuid4().hex[:6].lower()+"__"


def parse_proto_asm(proto_asm_file):
    op_table = {}
    curr_cell = None
    with open(proto_asm_file, 'r') as f:
        current_op = None
        for line in f:
            # remove all comments starting with #
            line = re.sub(r'#.*$', '', line)
            line = line.strip()
            if line == '':
                continue
            pattern = re.compile(r'^cell\s+(\w+)$')
            match = pattern.match(line)
            if match is not None:
                curr_cell = match.group(1)
                continue
            pattern = re.compile(r'^rop\s+(\w+)\s+(.+)$')
            match = pattern.match(line)
            if match is not None:
                if curr_cell is None:
                    logging.error('Resource operation definition must be inside a cell block!')
                    sys.exit(1)
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

                op = {'name': op_name, 'slot': slot, 'port': port, 'instr_list': [], 'cell': curr_cell}
                op_table[op['name']] = op
                current_op = op
                continue
            pattern = re.compile(r'^cop\s+(\w+)$')
            match = pattern.match(line)
            if match is not None:
                if curr_cell is None:
                    logging.error('Control operation definition must be inside a cell block!')
                    sys.exit(1)
                op_name = match.group(1)
                op = {'name': op_name, 'slot': -1, 'port': -1, 'instr_list': [], 'cell': curr_cell}
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

def extract_control_op_expr_block(op_name, instr_list, event_counter) -> list:
    expr = ""

    for instr in instr_list:
        instr = instr.strip()
    
    i=0
    while i < len(instr_list):
        instr = instr_list[i] 
        pattern = re.compile(r'looph\s+(.*)$')
        match = pattern.match(instr)
        if match is not None:
                iter = 0
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
                        if not field_value.isdigit():
                            logging.error('Invalid iter value: %s. Iteration must be a number!' % field_value)
                            sys.exit(1)
                        iter = int(field_value)
                loop_block_contents = []
                # add all instructions until reaching the last "loopt".
                for j in range(len(instr_list)-1, i, -1):
                    pattern = re.compile(r'loopt\s+(.*)$')
                    match = pattern.match(instr_list[j])
                    if match is not None:
                        # found the last "loopt"
                        for k in range(i+1, j+1):
                            loop_block_contents.append(instr_list[k])

                        expr = "T<"+generate_uuid()+">("+ expr + ", T<"+generate_uuid()+">("+op_name+"_e"+str(event_counter)+", R<"+str(iter)+", "+generate_uuid()+">("
                        event_counter += 1
                        e, event_counter = extract_control_op_expr_block(op_name, loop_block_contents, event_counter)
                        expr += e+")))"
                        i = j+1
                        break
                continue
        else:
            if expr == "":
                expr = op_name+"_e"+str(event_counter)
                event_counter += 1
            else:
                expr = "T<"+generate_uuid()+">(" + expr + ", "+op_name+"_e"+str(event_counter)+")"
                event_counter += 1
            i += 1
    return [expr, event_counter]

def extract_control_op_expr(op) -> str:
    e, counter = extract_control_op_expr_block(op['name'], op['instr_list'], 0)
    return e

def extract_resource_op_expr(op) -> str:
    # resource operation
        T={}
        R={}
        expr = op['name']+"_e0"
        for instr in op['instr_list']:
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

def extract_op_expr(op):
    if op['slot'] == -1:
        return extract_control_op_expr(op)
    else:
        return extract_resource_op_expr(op)

def parse_constraint(constraint_file):
    constraint_list = []
    curr_type = None
    with open(constraint_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            if line[0] == "#":
                continue
            pattern = re.compile(r'^type\s+(.+)$')
            match = pattern.match(line)
            if match is not None:
                curr_type = match.group(1)
                continue
            else:
                if curr_type is None:
                    logging.error('Undefined type for constraint: %s' % line)
                    sys.exit(1)
                constraint_list.append([curr_type, line])
    return constraint_list

def find_anchor_repeat_level(expr, event_name):
    # remove all white space of expr and event_name
    expr = re.sub(r'\s+', '', expr)
    event_name = re.sub(r'\s+', '', event_name)
    
    if expr != event_name:
        pattern = re.compile(r'^R<\d+,[^>]+>\((.+)\)$')
        match = pattern.match(expr)
        if match is not None:
            rl = find_anchor_repeat_level(match.group(1), event_name)
            if rl < 0:
                return -1
            else:
                return rl + 1
        pattern = re.compile(r'^T<[^>]+>\((.+)\)$')
        match = pattern.match(expr)
        if match is not None:
            operands = match.group(1)
            # it must have a comma to seperate the operands, find the comma that is not inside a pair of parentheses, or square brackets, or curly brackets, or angle brackets
            comma = -1
            parentheses = 0
            square_brackets = 0
            curly_brackets = 0
            angle_brackets = 0
            for i in range(len(operands)):
                if operands[i] == '(':
                    parentheses += 1
                if operands[i] == ')':
                    parentheses -= 1
                if operands[i] == '[':
                    square_brackets += 1
                if operands[i] == ']':
                    square_brackets -= 1
                if operands[i] == '{':
                    curly_brackets += 1
                if operands[i] == '}':
                    curly_brackets -= 1
                if operands[i] == '<':
                    angle_brackets += 1
                if operands[i] == '>':
                    angle_brackets -= 1
                if operands[i] == ',' and parentheses == 0 and square_brackets == 0 and curly_brackets == 0 and angle_brackets == 0:
                    comma = i
                    break
            if comma == -1:
                logging.error("Expression must have two operands: "+expr)
                exit(1)
            operands = [operands[:comma], operands[comma+1:]]
            if len(operands)!=2:
                logging.error("Expression must have two operands: "+expr)
                exit(1)
            rl = find_anchor_repeat_level(operands[0], event_name)
            if rl < 0:
                rl = find_anchor_repeat_level(operands[1], event_name)
                if rl < 0:
                    return -1
                else:
                    return rl
            else:
                return rl
        return -1
    return 0


def extract_anchor_symbols(expr, op_expr_table, symbol_table, anchor_table):
    symbols = []
    pattern = re.compile(r'([a-zA-Z_][\w]*)(\.e[0-9]+)?([\[\d\]]*)')
    matches = pattern.findall(expr)
    for match in matches:
        if match not in symbols:
            op_name = match[0]
            event_name = match[1]
            event_name_remove_dot = event_name
            if event_name != "":
                event_name_remove_dot = event_name[1:] # remove the dot
            else:
                event_name = ".e0"
                event_name_remove_dot = "e0"
            index_list = match[2]
            index_list_split = []
            if index_list != "":
                index_list_split = index_list[1:-1].split('][')
            
            full_match = op_name+event_name+index_list
            repeat_level = find_anchor_repeat_level(op_expr_table[op_name], op_name+"_"+event_name_remove_dot)
            if repeat_level < 0:
                logging.error('%s: Invalid anchor expression! Repeat level is %d!' % (full_match, repeat_level))
                sys.exit(1)
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
        pattern = re.compile(r'R<\d+,[^>]+>')
        matches = pattern.findall(expr)
        level = len(matches)
        repeat_level[op_name] = level
    return repeat_level

def find_all_anchor(expr) -> list:
    anchor_list = []

    counter = 0
    node_map = {}   
    root, counter  = operation.create_tree(node_map, counter, expr)
    
    def find_parent_nodes(root_node, target_value, parent_nodes)-> bool:
            if root_node is None:
                return False
            if root_node.value == target_value:
                return True
            if root_node.left is None and root_node.right is None:
                return False

            parent_nodes.append(root_node)
            if root_node.left is not None:
                ret = find_parent_nodes(root_node.left, target_value, parent_nodes)
                if ret:
                    return True
            
            if root_node.right is not None:
                ret = find_parent_nodes(root_node.right, target_value, parent_nodes)
                if ret:
                    return True
            parent_nodes.pop()
            
            return False
    
    for key in node_map:
        if isinstance(node_map[key], operation.Event):
            parent_nodes = []
            find_parent_nodes(root, key, parent_nodes)
            parent_R_nodes = [ node_map[x.value] for x in parent_nodes if isinstance(node_map[x.value], operation.RepeatitionOperator)]

            pattern = re.compile(r'([a-zA-Z_$][\w]*)_(e[0-9]+)')
            matches = pattern.match(node_map[key].name)
            if matches is not None:
                anchor_prefix = matches.group(1) + "." +matches.group(2)
            else:
                logging.error('Invalid event name: %s' % node_map[key].name)
                sys.exit(1)


            if len(parent_R_nodes) == 0:
                anchor_list.append(anchor_prefix)
            else:
                parent_nodes_iter = [x.iter for x in parent_R_nodes]
                all_permutation = [list(x) for x in itertools.product(*[range(i) for i in parent_nodes_iter])]
                for indices in all_permutation:
                    anchor = anchor_prefix
                    for i in range(len(indices)):
                        anchor = anchor + "[" + str(indices[i]) + "]"
                    anchor_list.append(anchor)

    return anchor_list


def add_build_in_constraints(op_table, op_expr_table, constraint_list):
    all_resource_op = []
    all_control_op_anchors = {}

    for op_name in op_table:
        op = op_table[op_name]
        if op['slot'] < 0:
            all_control_op_anchors[op['cell']] = []

    for op_name in op_table:
        op = op_table[op_name]
        if op['slot'] >=0:
            all_resource_op.append(op)
        else:
            all_control_op_anchors[op['cell']].extend(find_all_anchor(op_expr_table[op_name]))
    for i in range(len(all_resource_op)):
        op = all_resource_op[i]
        slot = int(op['slot'])
        port = int(op['port'])
        cell = op['cell']
        for j in range(i+1, len(all_resource_op)):
            op2 = all_resource_op[j]
            slot2 = int(op2['slot'])
            port2 = int(op2['port'])
            cell2 = op2['cell']
            if cell == cell2:
                if slot >= slot2+4 or slot2 >= slot+4:
                    if port != port2:
                        constraint_list.append(['linear', op['name']+" != "+op2['name']])
        if cell in all_control_op_anchors:
            for anchor in all_control_op_anchors[op['cell']]:
                constraint_list.append(['linear', op['name']+" != "+anchor])

def generate(proto_asm_file, constraint_file, output_dir):
    op_table = parse_proto_asm(proto_asm_file)
    op_expr_table = {}
    for op_name in op_table:
        op = op_table[op_name]
        expr = extract_op_expr(op)
        op_expr_table[op_name] = expr
    constraint_list = parse_constraint(constraint_file)
    add_build_in_constraints(op_table, op_expr_table, constraint_list)


    symbol_table = {}
    anchor_table = {}
    for i in range(len(constraint_list)):
        constraint = constraint_list[i][1]
        symbol_list = extract_anchor_symbols(constraint, op_expr_table, symbol_table, anchor_table)
        for symbol in symbol_list:
            constraint = constraint.replace(symbol, symbol_table[symbol])
        constraint_list[i][1] = constraint
    
    with open(os.path.join(output_dir, "model.txt"), 'w') as f:
        for op_name in op_expr_table:
            f.write("operation "+op_name+" "+op_expr_table[op_name]+"\n")
        for anchor in anchor_table:
            f.write("anchor "+anchor+" "+anchor_table[anchor]+"\n")
        for constraint in constraint_list:
            f.write("constraint "+constraint[0]+" "+constraint[1]+"\n")
    
    logging.info("Model file is generated at %s" % os.path.join(output_dir, "model.txt"))

    
