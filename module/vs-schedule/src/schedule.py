import os
import regex as re
import logging
import operation
import anchor
import constraint
import solver
import json
import trace
from sympy import simplify

def parse(file_path: str, op_table, anchor_dict, constraint_list, node_map, counter) -> None:
    with open(file_path, "r") as file:
        lines = file.readlines()
        counter = 0
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            if line[0] == "#":
                continue
            # match the pattern for operation definition
            pattern = re.compile(r"^operation\s+([a-zA-Z_\.$][\w$]*)\s+(.+)")
            if pattern.match(line):
                op_name = pattern.match(line).group(1)
                expr = pattern.match(line).group(2)
                counter = operation.add_operation(op_name, expr, op_table, node_map, counter)
                continue
            
            # match the pattern for anchor definition
            pattern = re.compile(r"^anchor\s+([a-zA-Z_\.$][\w$]*)\s+(.+)")
            if pattern.match(line):
                anchor_name = pattern.match(line).group(1)
                expr = pattern.match(line).group(2)
                anchor.add_anchor(anchor_name, expr, anchor_dict)
                continue

            # match the pattern for constraint definition
            pattern = re.compile(r"^constraint\s+([a-zA-Z_][\w]*)+\s+(.+)")
            if pattern.match(line):
                cstr_type = pattern.match(line).group(1)
                expr = pattern.match(line).group(2)
                constraint.add_constraint(cstr_type, expr, constraint_list)
                continue
            
            logging.error("Invalid line: "+line)
            exit(1)
    return counter


def schedule(model_file, output_dir):
    node_map = {}
    counter = 0
    op_table = {}
    anchor_dict = {}
    constraint_list = []
    counter = parse(model_file, op_table, anchor_dict, constraint_list, node_map, counter)
    [start_time_table, duration_table] =  operation.calculate_timing_properties(op_table, node_map)
    event_dict = operation.extract_event_dict(op_table, node_map)
    timing_variable_table = {}
    operation.add_var_to_timing_var_dict(op_table, node_map, start_time_table, timing_variable_table)
    anchor.add_anchor_to_timing_var_dict(anchor_dict, op_table, event_dict, node_map, duration_table, timing_variable_table)

    [latency, solution] = solver.solve(timing_variable_table, constraint_list, op_table, duration_table, 2**32-1)

    output_dict = {}
    output_dict["latency"] = latency
    output_dict["solution"] = solution

    logging.info("Schedule completed successfully!")
    logging.info("Solution:")
    
    # print output_dict as a pretty table
    solution_keys = list(solution.keys())
    solution_keys.sort()
    print("")
    print("+========================================================+")
    print("| {:<55}|".format("Latency="+str(latency)))
    print("+========================================================+")
    print("| {:<42} | {:<10}|".format('Variable','Value'))
    print("+--------------------------------------------+-----------+")
    for k in solution_keys:
        if not k.startswith("__"):
            print("| {:<42} | {:<10}|".format(k, solution[k]))
    print("+========================================================+")
    print("")

    with open(os.path.join(output_dir, "timing_table.json"), "w+") as file:
        json.dump(output_dict, file, indent=4)
    
    logging.info("Timing table saved to "+os.path.join(output_dir, "timing_table.json"))

    # create a trace file

    # use the solution to simplify the start_time_table and duration_table
    for op_name in start_time_table:
        stt = start_time_table[op_name]
        dt = duration_table[op_name]
        for key in stt:
            stt[key] = str(eval(stt[key], solution))
        for key in dt:
            dt[key] = str(eval(dt[key], solution))
    
    # use the solution to simplify the node_map nodes
    for key in node_map:
        if isinstance(node_map[key], operation.RepeatitionOperator):
            node_map[key].delay_var = str(eval(node_map[key].delay_var, solution))
        if isinstance(node_map[key], operation.TransitionOperator):
            node_map[key].delay_var = str(eval(node_map[key].delay_var, solution))

    all_events = []
    for op_name in op_table:
        event_list = trace.create_event_list_from_btree(op_table[op_name], op_name, solution[op_name], node_map, start_time_table[op_name], duration_table[op_name])
        all_events.extend(event_list)
    
    trace.create_trace_file_from_event_list(all_events, os.path.join(output_dir, "trace.json"))