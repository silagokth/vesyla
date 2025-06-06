#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import json
import re


def analyze_rep_instr(instr: str) -> list:
    # parse the instruction string and find out each fields
    instr = instr.strip()
    pattern = re.compile(r"rep\s+(.*)$")
    match = pattern.match(instr)
    if match is not None:
        level = 0
        iter = 1
        delay = 0
        step = 1
        fields = match.group(1).split(",")
        field_map = {}
        for field in fields:
            field = re.sub(r"\s+", "", field)
            pattern = re.compile(r"(\w+)\=(.+)")
            match = pattern.match(field)
            if match is None:
                logging.error("Invalid field format: %s" % field)
                sys.exit(1)
            field_name = match.group(1)
            field_value = match.group(2)
            if field_name == "iter":
                # check if iter is a number
                if not field_value.isdigit():
                    logging.error(
                        "Invalid iter value: %s. Iteration must be a number!" % iter
                    )
                    sys.exit(1)
                iter = int(field_value)
            if field_name == "step":
                step = int(field_value)
            elif field_name == "delay":
                delay = int(field_value)
            field_map[field_name] = field_value

        # check if iter, delay, step is equal or greater than 64
        flag_repx = False
        if iter > 64 or delay >= 64 or step >= 64:
            repx_iter = (iter - 1) // 64
            iter = (iter - 1) % 64 + 1
            repx_delay = delay // 64
            delay = delay % 64
            repx_step = step // 64
            step = step % 64
            flag_repx = True

        if flag_repx:
            # change field map
            field_map["iter"] = iter
            field_map["delay"] = delay
            field_map["step"] = step
            rep_instr = "rep " + ",".join(
                ["%s=%s" % (k, v) for k, v in field_map.items()]
            )
            field_map["iter"] = repx_iter
            field_map["delay"] = repx_delay
            field_map["step"] = repx_step
            repx_instr = "repx " + ",".join(
                ["%s=%s" % (k, v) for k, v in field_map.items()]
            )
            return [rep_instr, repx_instr]
        else:
            rep_instr = "rep " + ",".join(
                ["%s=%s" % (k, v) for k, v in field_map.items()]
            )
            return [rep_instr]
    return []


def replace_instr_field_variables(instr: str, time_table) -> str:
    # parse the instruction string:
    # it consists of a instruction name and a list of fields
    # each fields is separated by a comma, and the field name and value are separated by an equal sign
    instr = instr.strip()
    pattern = re.compile(r"^(\w+)$")
    match = pattern.match(instr)
    if match is not None:
        instr_name = match.group(1)
        instr = "%s" % instr_name
        return instr

    pattern = re.compile(r"^(\w+)\s+(.*)$")
    match = pattern.match(instr)
    if match is None:
        logging.error("Invalid instruction format: %s" % instr)
        sys.exit(1)
    instr_name = match.group(1)
    fields = match.group(2).split(",")
    for i in range(len(fields)):
        field = fields[i]
        field = re.sub(r"\s+", "", field)
        pattern = re.compile(r"(\w+)\=(.+)")
        match = pattern.match(field)
        if match is None:
            logging.error("Invalid field format: %s" % field)
            sys.exit(1)
        field_name = match.group(1)
        field_value = match.group(2)

        field_value = str(eval(field_value, time_table))

        fields[i] = "%s=%s" % (field_name, field_value)
    instr = "%s %s" % (instr_name, ",".join(fields))
    return instr


def create_time_table(timing_file: str):
    # the timing file is a json file. We only interested in the "solution" section which is a dictionary. The key is variable name and the value is the time.
    time_table = {}
    with open(timing_file, "r") as f:
        timing = json.load(f)
        time_table["__latency__"] = timing["latency"]
        for key, value in timing["solution"].items():
            time_table[key] = value
    return time_table


def create_operation_table(instr_file: str):
    op_table = {}
    with open(instr_file, "r") as f:
        current_cell = None
        current_op = None
        for line in f:
            # remove all comments starting with #
            line = re.sub(r"#.*$", "", line)
            line = line.lower()  # case insensitive
            line = line.strip()
            if line == "":
                continue
            pattern = re.compile(r"^cell\s+(\w+)$")
            match = pattern.match(line)
            if match is not None:
                cell_name = match.group(1)
                if cell_name not in op_table:
                    op_table[cell_name] = []
                current_cell = cell_name
                continue
            pattern = re.compile(r"^rop\s+(\w+)\s+(.+)$")
            match = pattern.match(line)
            if match is not None:
                op_name = match.group(1)
                fields = match.group(2).split(",")
                slot = 0
                port = 0
                for i in range(len(fields)):
                    fields[i] = re.sub(r"\s+", "", fields[i])
                    pattern = re.compile(r"(\w+)\=(.+)")
                    match = pattern.match(fields[i])
                    if match is None:
                        logging.error("Invalid field format: %s" % fields[i])
                        sys.exit(1)
                    field_name = match.group(1)
                    field_value = match.group(2)
                    if field_name == "slot":
                        slot = int(field_value)
                    elif field_name == "port":
                        port = int(field_value)
                    else:
                        logging.error("Invalid field name: %s" % field_name)
                        sys.exit(1)

                op = {
                    "name": op_name,
                    "slot": slot,
                    "port": port,
                    "instr_list": [],
                    "cell": current_cell,
                }
                op_table[current_cell].append(op)
                current_op = op
                continue
            pattern = re.compile(r"^cop\s+(\w+)$")
            match = pattern.match(line)
            if match is not None:
                op_name = match.group(1)
                slot = -1
                port = -1
                op = {
                    "name": op_name,
                    "slot": slot,
                    "port": port,
                    "instr_list": [],
                    "cell": current_cell,
                }
                op_table[current_cell].append(op)
                current_op = op
                continue
            pattern = re.compile(r"^(.+)$")
            match = pattern.match(line)
            if match is not None:
                instr = match.group(1)
                current_op["instr_list"].append(instr)
                continue
    return op_table


def create_act_instr(op_list):
    abs_port_idx = []
    for op in op_list:
        abs_port_idx.append(op["slot"] * 4 + op["port"])

    offset_slot = min(abs_port_idx) // 4
    relative_port_idx = [x - offset_slot * 4 for x in abs_port_idx]
    if max(relative_port_idx) < 16:
        # Mode 1 (continous) ACT instruction can activate all operations at the same time
        # converts the value in relative_port_idx to a 16-bit binary string
        bin_str = "".join(["1" if x in relative_port_idx else "0" for x in range(16)])
        # reverse the binary string since the port 0 is the least significant bit
        bin_str = bin_str[::-1]
        return "act mode=%d, param=%d, ports=0b%s" % (0, offset_slot, bin_str)

    slot_idx = []
    port_idx = []
    for op in op_list:
        slot_idx.append(op["slot"])
        port_idx.append(op["port"])
    if len(set(port_idx)) == 1:
        # Mode 2 (slot) ACT instruction can activate all operations in the same slot
        bin_str = "".join(["1" if x in slot_idx else "0" for x in range(4)])
        bin_str = bin_str[::-1]
        return "act mode=%d, param=%d, ports=0b%s" % (1, port_idx[0], bin_str)

    logging.error("Cannot create ACT instruction for operations: %s" % op_list)
    sys.exit(-1)
    return None


def insert_instr(instr_table: dict, op_table, time_table):
    for key in op_table:
        cell_instr_list = {}
        cell_timing_table = {}

        # First, insert instructions for control operations
        for op in op_table[key]:
            if op["slot"] >= 0:
                # this is a resource operation, ignore
                continue
            op_instr_list = op["instr_list"]
            for i in range(len(op_instr_list)):
                instr = op_instr_list[i]

        # create available time slots for each instruction position in cell_timing_table
        latency = time_table["__latency__"]
        clk = 0
        for i in range(latency):
            cell_timing_table[i] = clk
            clk += 1

        # insert control instructions
        for op in op_table[key]:
            if op["slot"] >= 0:
                # this is a resource operation, ignore
                continue
            op_instr_list = op["instr_list"]
            for i in range(len(op_instr_list)):
                instr = op_instr_list[i]
                cell_instr_list[
                    cell_timing_table[time_table[op["name"] + "_e" + str(i)]]
                ] = instr

        # insert the activation of all resource operations
        activation_dict = {}
        for op in op_table[key]:
            if op["slot"] < 0:
                # this is a control operation
                continue
            activation_time = time_table[op["name"]]
            if activation_time in activation_dict:
                activation_dict[activation_time].append(op)
            else:
                activation_dict[activation_time] = [op]

        for activation_time in activation_dict:
            op_list = activation_dict[activation_time]
            act_instr = create_act_instr(op_list)
            cell_instr_list[activation_time] = act_instr

        op_act_time = [x for x in activation_dict.keys()]
        op_act_time.sort(reverse=True)
        for t in op_act_time:
            op_list = activation_dict[t]
            for op in op_list:
                instr_list = op["instr_list"]
                # analyze the rep instruction
                new_instr_list = []
                for i in range(len(instr_list)):
                    instr = instr_list[i]
                    # check if the instruction is a rep instruction
                    instr = instr.strip()
                    if instr.startswith("rep"):
                        rep_instr = analyze_rep_instr(instr)
                        if len(rep_instr) > 0:
                            new_instr_list.extend(rep_instr)
                        else:
                            logging.error("Invalid rep instruction: %s" % instr)
                            sys.exit(1)
                    else:
                        new_instr_list.append(instr)
                instr_list = new_instr_list
                # reverse the order of the instructions
                instr_list = instr_list[::-1]
                tt = t - 1
                for instr in instr_list:
                    while True:
                        if tt in cell_instr_list:
                            tt -= 1
                        else:
                            cell_instr_list[tt] = instr
                            tt -= 1
                            break

        # now the instructions are inserted in the correct order
        instr_table[key] = cell_instr_list


def shift_instr_table(instr_table: dict):
    offset = 0

    for key in instr_table:
        if len(instr_table[key]) > 0:
            offset = min(offset, min(instr_table[key].keys()))

    offset = -offset
    shifted_instr_table = {}
    for key in instr_table:
        shifted_instr_table[key] = {}
        for t in instr_table[key]:
            shifted_instr_table[key][t + offset] = instr_table[key][t]
    return [shifted_instr_table, offset]


def insert_wait_instr(instr_table: dict, latency: int, op_table, time_table, offset):

    for key in instr_table:

        if key not in op_table:
            op_table[key] = []

        # First, insert instructions for control operations
        for op in op_table[key]:
            if op["slot"] >= 0:
                # this is a resource operation, ignore
                continue
            op_instr_list = op["instr_list"]
            for i in range(len(op_instr_list)):
                instr = op_instr_list[i]

        t = 0
        while t < latency:
            while t in instr_table[key] and t < latency:
                t = t + 1
            t_lb = t
            while t not in instr_table[key] and t < latency:
                t = t + 1
            t_ub = t
            cycles = t_ub - t_lb
            if cycles > 0:
                instr_table[key][t_lb] = "wait cycle=%d" % (cycles - 1)


def convert_instr_table_to_lists(instr_table: dict):
    instr_lists = {}
    for key in instr_table:
        instr_lists[key] = []
        ordered_list = [x for x in instr_table[key].keys()]
        ordered_list.sort()
        for t in ordered_list:
            instr_lists[key].append(instr_table[key][t])
    return instr_lists


def write_instr_lists_to_file(instr_lists: dict, output_folder: str):
    with open(os.path.join(output_folder, "0.asm"), "w+") as f:
        for key in instr_lists:
            f.write("cell %s\n" % key)
            for instr in instr_lists[key]:
                f.write("%s\n" % instr)
            f.write("\n")


def sync_instruction(instr_file: str, timing_file: str, cell_set) -> dict:
    time_table = create_time_table(timing_file)
    op_table = create_operation_table(instr_file)
    # check all the operations in the op_table has a corresponding timing in the time_table
    for key in op_table:
        for op in op_table[key]:
            if op["name"] not in time_table:
                logging.error(
                    "Operation %s does not have timing information" % op["name"]
                )
                return None

    for key in op_table:
        for op in op_table[key]:
            instr_list = op["instr_list"]
            for i in range(len(instr_list)):
                instr_list[i] = replace_instr_field_variables(instr_list[i], time_table)

    instr_table = {}
    for key in cell_set:
        instr_table[key] = {}
    latency = time_table["__latency__"]
    insert_instr(instr_table, op_table, time_table)
    instr_table, offset = shift_instr_table(instr_table)
    latency = latency + offset

    insert_wait_instr(instr_table, latency, op_table, time_table, offset)
    instr_lists = convert_instr_table_to_lists(instr_table)

    return instr_lists


def sync_resource(proto_asm_file, timing_file, cell_set, output_dir):
    instr_lists = sync_instruction(proto_asm_file, timing_file, cell_set)
    write_instr_lists_to_file(instr_lists, output_dir)
    logging.info("Instruction synchronization completed")
    logging.info(
        "Instruction lists are written to %s"
        % os.path.join(output_dir, "instr_lists.txt")
    )
