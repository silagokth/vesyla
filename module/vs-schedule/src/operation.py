from binarytree import Node

import regex as re
import logging

from sympy import simplify


class RepeatitionOperator:
    def __init__(self, iter: int, delay_var: str):
        self.iter = iter
        self.delay_var = delay_var
    
    def __str__(self):
        return "R<"+str(self.iter)+","+self.delay_var+">"

    def __repr__(self):
        return "R<"+str(self.iter)+","+self.delay_var+">"

class TransitionOperator:
    def __init__(self, delay_var: str):
        self.delay_var = delay_var
    
    def __str__(self):
        return "T<"+self.delay_var+">"
    
    def __repr__(self):
        return "T<"+self.delay_var+">"

class Event:
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

def add_node(expr, node_map, counter) -> int:
    node_map[counter] = expr
    counter += 1
    return counter

def create_tree(node_map, counter, expr: str) -> list:
    # remove all whitespaces, includeing tabs, newlines, etc.
    expr = re.sub(r"\s+", "", expr)
    logging.debug("Analyze expression: "+expr)

    # check if it matches the pattern for repeatition operator
    pattern = re.compile(r"^R<(\d+),([^>]+)>\((.+)\)$")
    match = pattern.match(expr)
    if match:
        # if it matches the pattern, then it is a repeatition operator
        iter = int(match.group(1))
        delay_var = match.group(2)
        expr = match.group(3)
        node = Node(counter)
        counter = add_node(RepeatitionOperator(iter, delay_var), node_map, counter)
        [node.left, counter] = create_tree(node_map, counter, expr)
        return [node, counter]
    
    # check if it matches the pattern for transition operator
    pattern = re.compile(r"^T<([^>]+)>\((.+)\)$")
    match = pattern.match(expr)
    if match:
        # if it matches the pattern, then it is a transition operator
        delay_var = match.group(1)
        operands = match.group(2)
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
        node = Node(counter)
        counter = add_node(TransitionOperator(delay_var), node_map, counter)
        [node.left, counter] = create_tree(node_map, counter, operands[0])
        [node.right, counter] = create_tree(node_map, counter, operands[1])
        return [node, counter]
    
    # check if it matches the pattern for event
    pattern = re.compile(r"^([a-zA-Z_\.$][\w$]*)$")
    match = pattern.match(expr)
    if match:
        # if it matches the pattern, then it is an event
        name = match.group(1)
        node = Node(counter)
        counter = add_node(Event(name), node_map, counter)
        return [node, counter]
    
    logging.error("Invalid expression: "+expr)
    exit(1)

def extract_event_dict_for_operation(op_name, op_table, node_map) -> dict:
    event_dict = {}
    node_ids = [ x for x in op_table[op_name].values if x is not None]
    for key in node_ids:
        if isinstance(node_map[key], Event):
            if node_map[key].name not in event_dict:
                event_dict[node_map[key].name] = key
            else:
                logging.error("Event "+node_map[key].name+" is duplicated")
                exit(1)
    return event_dict

def extract_event_dict(op_table,node_map) -> dict:
    event_dict = {}
    for op_name in op_table:
        node_ids = [ x for x in op_table[op_name].values if x is not None]
        for key in node_ids:
            if isinstance(node_map[key], Event):
                if node_map[key].name not in event_dict:
                    event_dict[node_map[key].name] = key
                else:
                    logging.error("Event "+node_map[key].name+" is duplicated")
                    exit(1)
    return event_dict

def extract_delay_expr_list(op_table, node_map) -> list:
    delay_var_expr_list = []
    for op_name in op_table:
        node_ids = [ x for x in op_table[op_name].values if x is not None]
        for key in node_ids:
            if isinstance(node_map[key], RepeatitionOperator):
                delay_var_expr_list.append(node_map[key].delay_var)
            if isinstance(node_map[key], TransitionOperator):
                delay_var_expr_list.append(node_map[key].delay_var)
    return delay_var_expr_list

def add_operation(op_name, expr, op_table, node_map, counter) -> int:
    if op_name in op_table:
        logging.error("Operation "+op_name+" is duplicated")
        exit(1)
    [op, counter] = create_tree(node_map, counter, expr)
    op_table[op_name] = op
    return counter

def left_middle_right_traversal(node: Node) -> list:
    traversal = []
    if node is not None:
        traversal = traversal + left_middle_right_traversal(node.left)
        traversal.append(node)
        traversal = traversal + left_middle_right_traversal(node.right)
    return traversal

def left_right_middle_traversal(node: Node) -> list:
    traversal = []
    if node is not None:
        traversal = traversal + left_right_middle_traversal(node.left)
        traversal = traversal + left_right_middle_traversal(node.right)
        traversal.append(node)
        
    return traversal

def middle_left_right_traversal(node: Node) -> list:
    traversal = []
    if node is not None:
        traversal.append(node)
        traversal = traversal + middle_left_right_traversal(node.left)
        traversal = traversal + middle_left_right_traversal(node.right)
    return traversal


def calculate_timing_properties(op_table, node_map) -> dict:
    start_time_table = {}
    duration_table = {}
    for op_name in op_table:
        
        start_dict = {}
        duration_dict = {}

        for x in op_table[op_name].values:
            if x is not None:
                start_dict[x] = "0"
                duration_dict[x] = "0"
        
        # do a left-right-middle traversal of the tree
        node = op_table[op_name]
        sequence = left_right_middle_traversal(node)

        for node in sequence:
            id = node.value
            if isinstance(node_map[id], Event):
                duration_dict[id] = "1"
            if isinstance(node_map[id], RepeatitionOperator):
                duration_dict[id] = "(" + duration_dict[node.left.value] + ") * (" + str(node_map[id].iter) + ") + (" + node_map[id].delay_var + ") * (" + str(node_map[id].iter) + " - 1)"
                duration_dict[id] = str(simplify(duration_dict[id]))
            if isinstance(node_map[id], TransitionOperator):
                duration_dict[id] = "(" + duration_dict[node.left.value] + ") + (" + node_map[id].delay_var + ") + (" + duration_dict[node.right.value] + ")"
                duration_dict[id] = str(simplify(duration_dict[id]))
        
        # do a middle-left-right traversal of the tree
        sequence = middle_left_right_traversal(node)
        start_dict[node.value] = "0"
        for node in sequence:
            if node.left is not None:
                start_dict[node.left.value] = str(simplify(start_dict[node.value]))
            if node.right is not None:
                start_dict[node.right.value] = str(simplify("(" + duration_dict[node.left.value] + ") + (" + start_dict[node.left.value] + ") + (" + node_map[node.value].delay_var + ")"))
        
        start_time_table[op_name] = start_dict
        duration_table[op_name] = duration_dict
        
    return [start_time_table, duration_table]

def add_var_to_timing_var_dict(op_table, node_map, start_time_table, timing_var_dict):
    for op_name in op_table:
        if op_name in timing_var_dict:
            logging.error("Operation "+op_name+" is duplicated")
            exit(1)
        timing_var_dict[op_name] = op_name

        event_dict = extract_event_dict_for_operation(op_name, op_table, node_map)
        for event_name in event_dict:
            if event_name in timing_var_dict:
                logging.error("Event "+event_name+" is duplicated")
                exit(1)
            timing_var_dict[event_name] = str(simplify("("+op_name+")+("+start_time_table[op_name][event_dict[event_name]]+")"))

    for delay_var_expr in extract_delay_expr_list(op_table, node_map):
        symbols = list(simplify(delay_var_expr).free_symbols)
        for symbol in symbols:
            if symbol.name not in timing_var_dict:
                timing_var_dict[symbol.name] = symbol.name