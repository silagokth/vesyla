import re
import logging
from sympy import simplify
import operation


def add_anchor(name: str, expr: str, anchor_dict: dict):
    if name in anchor_dict:
        logging.error(
            "Name " + name + " is already defined, choose another name for the anchor"
        )
        exit(1)
    anchor_dict[name] = expr


def add_anchor_to_timing_var_dict(
    anchor_dict: dict,
    op_table,
    event_dict_table,
    node_map,
    duration_table,
    timing_var_dict,
):
    for name in anchor_dict:
        expr = anchor_dict[name]
        if name in timing_var_dict:
            logging.error(
                "Name "
                + name
                + " is already defined, choose another name for the anchor"
            )
            exit(1)
        timing_var_dict[name] = slice_name_translation(
            expr, op_table, event_dict_table, node_map, duration_table, timing_var_dict
        )


def slice_name_translation(
    slice_name: str,
    op_table,
    event_dict_table,
    node_map,
    duration_table,
    timing_var_dict,
) -> str:
    slice_name = re.sub(r"\s+", "", slice_name)

    # match the pattern of slice name: it must have two parts
    # part 1 is the name of the event or the delay variable, it must be a valid identifier
    # part 2 is the slices list. The list includes a list of integers, each integer is wrapped by a pair of square brackets. It must contain at least one integer
    # no need to consider whitespace, as it is already removed

    pattern = re.compile(r"([a-zA-Z_$][\w$]*)((\[\d+\])+)$")
    match = pattern.match(slice_name)
    if not match:
        logging.error("Invalid slice name: " + slice_name)
        exit(1)

    # get the event or delay variable name
    name = match.group(1)
    # get the slices list
    slices = match.group(2)
    # extract the integers from the slices list
    slices_list = []
    # break down slices into a list of integers. Slices are integers. Each of them is wrapped by a pair of square brackets. First get a list of strings that start with a square bracket and end with a square bracket
    if slices is not None:
        slices_list = [int(x) for x in re.findall(r"\[(\d+)\]", slices)]
        # remove the square brackets
        for i in range(len(slices_list)):
            slices_list[i] = int(slices_list[i])

    if name not in event_dict_table:
        logging.error("Event or delay variable " + name + " is not defined")
        exit(1)

    # find op_name of the event
    op_name = None
    for key in op_table:
        if event_dict_table[name] in op_table[key].values:
            op_name = key
            break

    if op_name is None:
        logging.error("Event " + name + " is not in any operation")
        exit(1)

    # find out each R operator outside the event
    parent_nodes = []
    root_node = op_table[op_name]

    # find all parent nodes that leads to the event
    def find_parent_nodes(root_node, target_value, parent_nodes) -> bool:
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

    find_parent_nodes(root_node, event_dict_table[name], parent_nodes)
    parent_R_nodes = [
        x
        for x in parent_nodes
        if isinstance(node_map[x.value], operation.RepeatitionOperator)
    ]

    if len(parent_R_nodes) != len(slices_list):
        logging.error(
            "The number of slices in "
            + slice_name
            + " does not match the number of R operators outside the event"
        )
        exit(1)

    expr = timing_var_dict[name]
    for i in range(len(slices_list)):
        slice = slices_list[i]
        iter = node_map[parent_R_nodes[i].value].iter
        delay_var = node_map[parent_R_nodes[i].value].delay_var
        if slice >= iter:
            logging.error(
                "The Slice "
                + str(i)
                + " in "
                + name
                + " (="
                + str(slice)
                + ") is out of range"
            )
            exit(1)
        expr = str(
            simplify(
                expr
                + " + (("
                + duration_table[op_name][parent_R_nodes[i].left.value]
                + ")+"
                + delay_var
                + ") * "
                + str(slice)
            )
        )
    return expr
