import sys
import argparse
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
            pattern = re.compile(r"^operation\s+([a-zA-Z_$][\w$]*)\s+(.+)")
            if pattern.match(line):
                op_name = pattern.match(line).group(1)
                expr = pattern.match(line).group(2)
                counter = operation.add_operation(op_name, expr, op_table, node_map, counter)
                continue
            
            # match the pattern for anchor definition
            pattern = re.compile(r"^anchor\s+([a-zA-Z_$][\w$]*)\s+(.+)")
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



def main(args):
    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s' )

    # parse the arguments
    parser = argparse.ArgumentParser(description="vs-schedule")
    # -i/--input: the input file
    parser.add_argument("-i", "--input", help="The input file", required=True)
    # -o/--output: the output file
    parser.add_argument("-o", "--output", help="The output file", required=True)
    arg_list = parser.parse_args(args)


    node_map = {}
    counter = 0
    op_table = {}
    anchor_dict = {}
    constraint_list = []
    counter = parse(arg_list.input, op_table, anchor_dict, constraint_list, node_map, counter)
    [start_time_table, duration_table] =  operation.calculate_timing_properties(op_table, node_map)
    event_dict = operation.extract_event_dict(op_table, node_map)
    timing_variable_table = {}
    operation.add_var_to_timing_var_dict(op_table, node_map, start_time_table, timing_variable_table)
    anchor.add_anchor_to_timing_var_dict(anchor_dict, op_table, event_dict, node_map, duration_table, timing_variable_table)

    [latency, solution] = solver.solve(timing_variable_table, constraint_list, op_table, duration_table, 10000)

    output_dict = {}
    output_dict["latency"] = latency
    output_dict["solution"] = solution

    logging.info("Schedule completed successfully!")
    logging.info("Solution:")
    
    # print output_dict as a pretty table
    print("")
    print("+============================+")
    print("| {:<27}|".format("Latency="+str(latency)))
    print("+============================+")
    print("| {:<14} | {:<10}|".format('Variable','Value'))
    print("+----------------+-----------+")
    for k, v in solution.items():
        print("| {:<14} | {:<10}|".format(k, v))
    print("+============================+")
    print("")

    with open(arg_list.output, "w") as file:
        json.dump(output_dict, file, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])