import os
import regex as re
import logging
import operation
import anchor
import constraint
import solver
import json

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
            pattern = re.compile(r"^constraint\s+(.+)")
            if pattern.match(line):
                expr = pattern.match(line).group(1)
                constraint.add_constraint(expr, constraint_list)
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

    [latency, solution] = solver.solve(timing_variable_table, constraint_list, op_table, duration_table, 20000)

    output_dict = {}
    output_dict["latency"] = latency
    output_dict["solution"] = solution

    logging.info("Schedule completed successfully!")
    logging.info("Solution:")
    
    # print output_dict as a pretty table
    print("")
    print("+========================================================+")
    print("| {:<55}|".format("Latency="+str(latency)))
    print("+========================================================+")
    print("| {:<42} | {:<10}|".format('Variable','Value'))
    print("+--------------------------------------------+-----------+")
    for k, v in solution.items():
        print("| {:<42} | {:<10}|".format(k, v))
    print("+========================================================+")
    print("")

    with open(os.path.join(output_dir, "timing_table.json"), "w+") as file:
        json.dump(output_dict, file, indent=4)
    
    logging.info("Timing table saved to "+os.path.join(output_dir, "timing_table.json"))