import os
import sys
import re
import copy

import DataStructure_pb2 as ds
import verboselogs

logger = verboselogs.VerboseLogger('vesim')

def partial_update_variable(variable, value, value_bitwidth, low_starting_bit) -> int:
    '''update a part of a variable with a new value'''
    if value_bitwidth <= 0:
        logging.error("value_bitwidth must be greater than 0")
        exit(1)
    if low_starting_bit < 0:
        logging.error("low_starting_bit must be greater than or equal to 0")
        exit(1)
    mask = int('1' * value_bitwidth, 2)
    mask = mask << low_starting_bit
    mask = ~mask
    variable = variable & mask
    # trunkate value to the value_bitwidth, remove all bits higher than value_bitwidth
    value = value & int('1' * value_bitwidth, 2)
    value = value << low_starting_bit
    variable = variable | value
    return variable


def resource_trigger(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    db = resource_pool_.get("db")
    slot = args[2]
    port = args[3]
    prefix = str(i)+"_"+str(j)+"_"+str(slot)
    resource_map = resource_pool_.get("resource_map")
    resource_name = resource_map[prefix]
    if resource_name == "rf":
        if port == 0:
            handler_name = "write_word"
        elif port == 1:
            handler_name = "read_word"
        elif port == 2:
            handler_name = "write_bulk"
        elif port == 3:
            handler_name = "read_bulk"
        else:
            logger.error("Error: Unknown port: ", port)
            return False
        conf = resource_pool_.get("conf")
        resource_conf = conf["rf_{}_{}_{}_{}_conf".format(i, j, slot, port)]
        storage_map = resource_pool_.get("storage_map")
        storage_resource = storage_map["rf_{}_{}_{}_{}_conf".format(
            i, j, slot, port)]
        levels = resource_conf["repeat"]
        t = clk_
        for l7 in range(levels[7]["iter"]):
            addr7 = l7*levels[7]["step"]
            for l6 in range(levels[6]["iter"]):
                addr6 = l6*levels[6]["step"]
                for l5 in range(levels[5]["iter"]):
                    addr5 = l5*levels[5]["step"]
                    for l4 in range(levels[4]["iter"]):
                        addr4 = l4*levels[4]["step"]
                        for l3 in range(levels[3]["iter"]):
                            addr3 = l3*levels[3]["step"]
                            for l2 in range(levels[2]["iter"]):
                                addr2 = l2*levels[2]["step"]
                                for l1 in range(levels[1]["iter"]):
                                    addr1 = l1*levels[1]["step"]
                                    for l0 in range(levels[0]["iter"]):
                                        addr0 = l0*levels[0]["step"]
                                        addr = resource_conf["start"] + addr7 + addr6 + \
                                            addr5+addr4+addr3+addr2+addr1+addr0
                                        if handler_name.startswith("write"):
                                            e = (t, handler_name, [
                                                i, j, slot, port, storage_resource, addr], 20, True)
                                        else:
                                            e = (t, handler_name, [
                                                i, j, slot, port, storage_resource, addr], 80, True)
                                        event_pool_.post(e)
                                        t += 1
                                        if l0 < levels[0]["iter"]-1:
                                            t += levels[0]["delay"]
                                    if l1 < levels[1]["iter"]-1:
                                        t += levels[1]["delay"]
                                if l2 < levels[2]["iter"]-1:
                                    t += levels[2]["delay"]
                            if l3 < levels[3]["iter"]-1:
                                t += levels[3]["delay"]
                        if l4 < levels[4]["iter"]-1:
                            t += levels[4]["delay"]
                    if l5 < levels[5]["iter"]-1:
                        t += levels[5]["delay"]
                if l6 < levels[6]["iter"]-1:
                    t += levels[6]["delay"]
            if l7 < levels[7]["iter"]-1:
                t += levels[7]["delay"]
        conf["rf_{}_{}_{}_{}_conf".format(i, j, slot, port)] = {"start": 0, "repeat": [
            {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        resource_pool_.set("conf", conf)
    elif resource_name == 'dpu':
        handler_name = "set_curr_dpu_mode"
        conf = resource_pool_.get("conf")
        resource_conf = conf["dpu_{}_{}_{}_conf".format(i, j, slot)]
        levels = resource_conf["repeat"]
        t = clk_
        for l7 in range(levels[7]["iter"]):
            addr7 = l7*levels[7]["step"]
            for l6 in range(levels[6]["iter"]):
                addr6 = l6*levels[6]["step"]
                for l5 in range(levels[5]["iter"]):
                    addr5 = l5*levels[5]["step"]
                    for l4 in range(levels[4]["iter"]):
                        addr4 = l4*levels[4]["step"]
                        for l3 in range(levels[3]["iter"]):
                            addr3 = l3*levels[3]["step"]
                            for l2 in range(levels[2]["iter"]):
                                addr2 = l2*levels[2]["step"]
                                for l1 in range(levels[1]["iter"]):
                                    addr1 = l1*levels[1]["step"]
                                    for l0 in range(levels[0]["iter"]):
                                        addr0 = l0*levels[0]["step"]
                                        addr = addr7 + addr6 + \
                                            addr5+addr4+addr3+addr2+addr1+addr0
                                        for x in range(resource_conf["max_state"]+1):
                                            print(
                                                "set option {} at time {}".format(x, t))
                                            dpuconf = resource_conf["option"][x]
                                            mode = dpuconf["mode"]
                                            imm = dpuconf["imm"]
                                            e = (t, handler_name, [
                                                i, j, slot, mode, imm], 100, True)
                                            event_pool_.post(e)
                                            t += 1
                                            if x < resource_conf["max_state"]:
                                                t += resource_conf["delay"][x]
                                        if l0 < levels[0]["iter"]-1:
                                            t += levels[0]["delay"]
                                    if l1 < levels[1]["iter"]-1:
                                        t += levels[1]["delay"]
                                if l2 < levels[2]["iter"]-1:
                                    t += levels[2]["delay"]
                            if l3 < levels[3]["iter"]-1:
                                t += levels[3]["delay"]
                        if l4 < levels[4]["iter"]-1:
                            t += levels[4]["delay"]
                    if l5 < levels[5]["iter"]-1:
                        t += levels[5]["delay"]
                if l6 < levels[6]["iter"]-1:
                    t += levels[6]["delay"]
            if l7 < levels[7]["iter"]-1:
                t += levels[7]["delay"]
        conf["dpu_{}_{}_{}_conf".format(i, j, slot)] = {
            "option": [{} for i in range(4)], "delay": [0 for i in range(3)], "max_state": 0, "repeat": [{"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        resource_pool_.set("conf", conf)
    elif resource_name == 'swb':
        if port == 0:
            handler_name = "set_curr_swb_mode"
        elif port == 2:
            handler_name = "set_curr_route_mode"
        else:
            logger.error("Error: Unknown port for SWB/ROUTE resource: ", port)
            return False
        conf = resource_pool_.get("conf")
        resource_conf = conf["swb_{}_{}_{}_{}_conf".format(i, j, slot, port)]
        levels = resource_conf["repeat"]
        t = clk_
        for l7 in range(levels[7]["iter"]):
            addr7 = l7*levels[7]["step"]
            for l6 in range(levels[6]["iter"]):
                addr6 = l6*levels[6]["step"]
                for l5 in range(levels[5]["iter"]):
                    addr5 = l5*levels[5]["step"]
                    for l4 in range(levels[4]["iter"]):
                        addr4 = l4*levels[4]["step"]
                        for l3 in range(levels[3]["iter"]):
                            addr3 = l3*levels[3]["step"]
                            for l2 in range(levels[2]["iter"]):
                                addr2 = l2*levels[2]["step"]
                                for l1 in range(levels[1]["iter"]):
                                    addr1 = l1*levels[1]["step"]
                                    for l0 in range(levels[0]["iter"]):
                                        addr0 = l0*levels[0]["step"]
                                        addr = addr7 + addr6 + \
                                            addr5+addr4+addr3+addr2+addr1+addr0
                                        for x in range(resource_conf["max_state"]+1):
                                            print(
                                                "set option {} at time {}".format(x, t))
                                            conn = resource_conf["option"][x]
                                            e = (t, handler_name, [
                                                i, j, slot, conn], 100, True)
                                            event_pool_.post(e)
                                            t += 1
                                            if x < resource_conf["max_state"]:
                                                t += resource_conf["delay"][x]
                                        if l0 < levels[0]["iter"]-1:
                                            t += levels[0]["delay"]
                                    if l1 < levels[1]["iter"]-1:
                                        t += levels[1]["delay"]
                                if l2 < levels[2]["iter"]-1:
                                    t += levels[2]["delay"]
                            if l3 < levels[3]["iter"]-1:
                                t += levels[3]["delay"]
                        if l4 < levels[4]["iter"]-1:
                            t += levels[4]["delay"]
                    if l5 < levels[5]["iter"]-1:
                        t += levels[5]["delay"]
                if l6 < levels[6]["iter"]-1:
                    t += levels[6]["delay"]
            if l7 < levels[7]["iter"]-1:
                t += levels[7]["delay"]
        conf["swb_{}_{}_{}_{}_conf".format(i, j, slot, port)] = {
            "option": [{} for i in range(4)], "delay": [0 for i in range(3)], "max_state": 0, "repeat": [{"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        resource_pool_.set("conf", conf)
    elif resource_name == 'iosram':
        conf = resource_pool_.get("conf")
        resource_conf = conf["iosram_{}_{}_{}_{}_conf".format(
            i, j, slot, port)]
        storage_map = resource_pool_.get("storage_map")
        storage_resource = storage_map["iosram_{}_{}_{}_{}_conf".format(
            i, j, slot, port)]
        storage_resource_start_slot = int(storage_resource.split("_")[-2])

        if slot == storage_resource_start_slot:
            if port == 0:
                handler_name = "read_from_io"
            elif port == 1:
                handler_name = "write_to_io"
            elif port == 2:
                handler_name = "io_write_to_sram"
            elif port == 3:
                handler_name = "sram_read_to_io"
            else:
                logger.error("Error: Unknown port for IOSRAM resource: ", port)
                return False
        else:
            if port == 2:
                handler_name = "write_bulk"
            elif port == 3:
                handler_name = "read_bulk"
            else:
                logger.error("Error: Unknown port for IOSRAM resource: ", port)
                return False

        levels = resource_conf["repeat"]
        t = clk_
        int_addr = 0
        for l7 in range(levels[7]["iter"]):
            addr7 = l7*levels[7]["step"]
            for l6 in range(levels[6]["iter"]):
                addr6 = l6*levels[6]["step"]
                for l5 in range(levels[5]["iter"]):
                    addr5 = l5*levels[5]["step"]
                    for l4 in range(levels[4]["iter"]):
                        addr4 = l4*levels[4]["step"]
                        for l3 in range(levels[3]["iter"]):
                            addr3 = l3*levels[3]["step"]
                            for l2 in range(levels[2]["iter"]):
                                addr2 = l2*levels[2]["step"]
                                for l1 in range(levels[1]["iter"]):
                                    addr1 = l1*levels[1]["step"]
                                    for l0 in range(levels[0]["iter"]):
                                        addr0 = l0*levels[0]["step"]
                                        addr = addr7 + addr6 + \
                                            addr5+addr4+addr3+addr2+addr1+addr0
                                        if slot == storage_resource_start_slot:
                                            if port == 0 or port == 3:
                                            # make it very low priority so that the updated data will not be immediately used
                                                e = (t, handler_name, [
                                                i, j, slot, storage_resource, addr + resource_conf["start"]], 10, True)
                                            elif port == 1 or port == 2:
                                                e = (t, handler_name, [
                                                i, j, slot, storage_resource, addr + resource_conf["start"]], 5, True)
                                            else:
                                                logger.error("Error: Unknown port for IOSRAM resource: ", port)
                                                return False
                                        else:
                                            if port == 2:
                                                if handler_name.startswith("write"):
                                                    e = (t, handler_name, [
                                                    i, j, slot, port, storage_resource, addr+resource_conf["start"]], 20, True)
                                            elif port == 3:
                                                e = (t, handler_name, [
                                                    i, j, slot, port, storage_resource, addr+resource_conf["start"]], 80, True)
                                            else:
                                                logger.error("Error: Unknown port for IOSRAM resource: ", port)
                                                return False
                                        event_pool_.post(e)
                                        int_addr += 1
                                        t += 1
                                        if l0 < levels[0]["iter"]-1:
                                            t += levels[0]["delay"]
                                    if l1 < levels[1]["iter"]-1:
                                        t += levels[1]["delay"]
                                if l2 < levels[2]["iter"]-1:
                                    t += levels[2]["delay"]
                            if l3 < levels[3]["iter"]-1:
                                t += levels[3]["delay"]
                        if l4 < levels[4]["iter"]-1:
                            t += levels[4]["delay"]
                    if l5 < levels[5]["iter"]-1:
                        t += levels[5]["delay"]
                if l6 < levels[6]["iter"]-1:
                    t += levels[6]["delay"]
            if l7 < levels[7]["iter"]-1:
                t += levels[7]["delay"]
        conf["iosram_{}_{}_{}_{}_conf".format(i, j, slot, port)] = {"start": 0, "repeat": [
            {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        resource_pool_.set("conf", conf)
    return True


def delay_signal(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    slot = args[2]
    port = args[3]
    signal_value = args[4]
    curr_value = resource_pool_.get("curr_value")
    curr_value[i][j][slot][port] = signal_value
    resource_pool_.set("curr_value", curr_value)
    return True


def write_word(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    slot = args[2]
    port = args[3]
    resource_name = args[4]
    addr = args[5]

    curr_value = resource_pool_.get("curr_value")
    value = curr_value[i][j][slot][port]

    if hasattr(value, "__len__"):
        logger.error("Error: value must be a single word")
        exit(-1)

    regs = resource_pool_.get(resource_name)
    regs[addr] = value
    resource_pool_.set(resource_name, regs)
    logger.info("Write word: "+str(value) +
                " to {}[{}]".format(resource_name, addr))
    return True


def read_word(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    slot = args[2]
    port = args[3]
    resource_name = args[4]
    addr = args[5]

    regs = resource_pool_.get(resource_name)
    value = regs[addr]

    if hasattr(value, "__len__"):
        logger.error("Error: value must be a single word")
        exit(-1)

    e = (clk_+1, "delay_signal", [i, j, slot, port, value], 100, True)
    event_pool_.post(e)
    logger.info("Read word: "+str(value) +
                " from {}[{}]".format(resource_name, addr))
    return True


def write_bulk(clk_, event_pool_, resource_pool_, handler_pool_, args):
    logger.debug("write_bulk is triggered: "+str(args))
    conneciton_map = resource_pool_.get("connection_map")
    print(conneciton_map)
    i = args[0]
    j = args[1]
    slot = args[2]
    port = args[3]
    resource_name = args[4]
    addr = args[5]

    curr_value = resource_pool_.get("curr_value")
    value = curr_value[i][j][slot][port]

    if not hasattr(value, "__len__") or len(value) != 16:
        logger.error("Error: Bulk write value length is not 16")
        exit(-1)

    if resource_name.startswith("rf"):
        regs = resource_pool_.get(resource_name)
        regs[addr*16:(addr+1)*16] = value
    else:
        regs = resource_pool_.get(resource_name)
        regs[addr] = value
    resource_pool_.set(resource_name, regs)
    logger.info("Write bulk: "+str(value) +
                " to {}[{}]".format(resource_name, addr))
    return True


def read_bulk(clk_, event_pool_, resource_pool_, handler_pool_, args):
    logger.debug("read_bulk is triggered: "+str(args))
    i = args[0]
    j = args[1]
    slot = args[2]
    port = args[3]
    resource_name = args[4]
    addr = args[5]

    if resource_name.startswith("rf"):
        regs = resource_pool_.get(resource_name)
        value = regs[addr*16:(addr+1)*16]
    else:
        regs = resource_pool_.get(resource_name)
        value = regs[addr]
    
    if not hasattr(value, "__len__") or len(value) != 16:
        logger.error("Error: Bulk read value length is not 16")
        exit(-1)

    e = (clk_+1, "delay_signal", [i, j, slot, port, value], 100, True)
    event_pool_.post(e)
    logger.info("Read bulk: "+str(value) +
                " from {}[{}]".format(resource_name, addr))
    return True


def read_from_io(clk_, event_pool_, resource_pool_, handler_pool_, args):
    logger.debug("read_from_io is triggered: "+str(args))
    i = args[0]
    j = args[1]
    slot = args[2]
    resource_name = args[3]
    addr = args[4]

    input_buffer = resource_pool_.get("input_buffer")
    io_temp_var = resource_pool_.get("io_temp_var")
    print(i, j, slot)
    io_temp_var["input2sram"]["{}_{}_{}".format(i,j,slot)] = input_buffer[addr]
    resource_pool_.set("io_temp_var", io_temp_var)

    iap = resource_pool_.get("IAP")
    iap.append([clk_, addr])
    resource_pool_.set("IAP", iap)

    logger.info("Read from input buffer: "+str(input_buffer[addr]))
    return True

def io_write_to_sram(clk_, event_pool_, resource_pool_, handler_pool_, args):
    logger.debug("io_write_to_sram is triggered: "+str(args))
    i = args[0]
    j = args[1]
    slot = args[2]
    resource_name = args[3]
    addr = args[4]

    sram = resource_pool_.get(resource_name)
    io_temp_var = resource_pool_.get("io_temp_var")
    sram[addr] = io_temp_var["input2sram"]["{}_{}_{}".format(i,j,slot)]
    resource_pool_.set(resource_name, sram)

    logger.info("Write to SRAM: "+str(sram[addr]))
    return True



def write_to_io(clk_, event_pool_, resource_pool_, handler_pool_, args):
    logger.debug("write_to_io is triggered: "+str(args))
    i = args[0]
    j = args[1]
    slot = args[2]
    resource_name = args[3]
    addr = args[4]

    output_buffer = resource_pool_.get("output_buffer")
    io_temp_var = resource_pool_.get("io_temp_var")
    output_buffer[addr] = io_temp_var["sram2output"]["{}_{}_{}".format(i,j,slot)]
    resource_pool_.set("output_buffer", output_buffer)

    output_buffer_active = resource_pool_.get("output_buffer_active")
    output_buffer_active[addr] = True
    resource_pool_.set("output_buffer_active", output_buffer_active)

    oap = resource_pool_.get("OAP")
    oap.append([clk_, addr])
    resource_pool_.set("OAP", oap)

    logger.info("Write to output buffer: "+str(output_buffer[addr]))
    return True

def sram_read_to_io(clk_, event_pool_, resource_pool_, handler_pool_, args):
    logger.debug("sram_read_to_io is triggered: "+str(args))
    i = args[0]
    j = args[1]
    slot = args[2]
    resource_name = args[3]
    addr = args[4]

    sram = resource_pool_.get(resource_name)
    io_temp_var = resource_pool_.get("io_temp_var")
    io_temp_var["sram2output"]["{}_{}_{}".format(i,j,slot)] = sram[addr]
    resource_pool_.set("io_temp_var", io_temp_var)

    logger.info("Read from SRAM: "+str(sram[addr]))
    return True


def set_current_dpu_mode(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    slot = args[2]
    mode = args[3]
    imm = args[4]

    curr_dpu_mode = resource_pool_.get("curr_dpu_mode")
    curr_dpu_mode["{}_{}_{}".format(i, j, slot)] = {
        "active": True, "mode": mode, "imm": imm}
    resource_pool_.set("curr_dpu_mode", curr_dpu_mode)
    logger.info("Set current DPU mode: "+str(mode)+" to " +
                str(i)+"_"+str(j)+"_"+str(slot))
    return True


def set_current_swb_mode(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    slot = args[2]
    conn = args[3]

    connection_map = resource_pool_.get("connection_map")
    connection_map["{}_{}_{}".format(i, j, slot)]["word"] = conn
    resource_pool_.set("connection_map", connection_map)
    logger.info("Set current SWB mode at " +
                str(i)+"_"+str(j)+"_"+str(slot))
    return True


def set_current_route_mode(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    slot = args[2]
    conn = args[3]

    connection_map = resource_pool_.get("connection_map")
    connection_map["{}_{}_{}".format(i, j, slot)]["bulk"] = conn
    resource_pool_.set("connection_map", connection_map)
    logger.info("Set current ROUTE mode at " +
                str(i)+"_"+str(j)+"_"+str(slot))
    return True


def compute_npc(resource_pool_, row_, col_, pc_, pc_increment_):
    # loop_manager = resource_pool_.get("loop_manager")
    # for i in range(4):
    #     curr_loop_id = 3-i
    #     loop_config = loop_manager[row_][col_][curr_loop_id]
    #     if loop_config["enable"]:
    #         if pc_ == loop_config["pc_end"]:
    #             if loop_config["iter"] <= 1:
    #                 loop_config["enable"] = False
    #                 loop_manager[row_][col_][curr_loop_id] = loop_config
    #                 resource_pool_.set("loop_manager", loop_manager)
    #                 continue
    #             else:
    #                 loop_config["iter"] = loop_config["iter"] - 1
    #                 raccu_reg = resource_pool_.get("raccu_reg")
    #                 raccu_reg[row_][col_][15 -
    #                                       curr_loop_id] += loop_config["step"]
    #                 resource_pool_.set("raccu_reg", raccu_reg)
    #                 loop_manager[row_][col_][curr_loop_id] = loop_config
    #                 resource_pool_.set("loop_manager", loop_manager)
    #                 return loop_config["pc_start"]
    #         else:
    #             return pc_ + pc_increment_
    #     else:
    #         continue

    return pc_+pc_increment_


def get_instr_field(instr: ds.Instruction, field_name: str) -> (bool, int):
    for field in instr.value_map:
        if field.key == field_name:
            return (True, field.val)
    return (False, 0)


def is_resource_instr(name: str, isa: ds.InstructionSet) -> bool:
    for instr in isa.instruction_templates:
        if instr.name == name:
            code = instr.code
            string_format = '{0:0'+str(isa.instr_code_bitwidth)+'b}'
            code_bin = string_format.format(code)
            if code_bin[0] == '0':
                return False
            else:
                return True
    return False


def calc_regs(clk_, event_pool_, resource_pool_, handler_pool_, args):
    print(args)
    row = args[0]
    col = args[1]
    mode = args[2]
    operand1_sd = args[3]
    operand1 = args[4]
    operand2_sd = args[5]
    operand2 = args[6]
    result = args[7]
    if operand2 >= 2**9:
        operand2 = operand2 - 2**10
    if operand1 >= 2**9:
        operand1 = operand1 - 2**10

    raccu_reg = resource_pool_.get("raccu_reg")
    if operand1_sd:
        operand1 = raccu_reg[row][col][operand1]
    if operand2_sd:
        operand2 = raccu_reg[row][col][operand2]

    if mode == 1:
        raccu_reg[row][col][result] = operand1 + operand2
    elif mode == 2:
        raccu_reg[row][col][result] = operand1 - operand2
    elif mode == 3:
        raccu_reg[row][col][result] = operand1 // (2**operand2)
    elif mode == 4:
        raccu_reg[row][col][result] = operand1 * (2**operand2)
    elif mode == 5:
        raccu_reg[row][col][result] = operand1 * operand2
    else:
        logger.fatal("Unknown RACCU mode: " + str(mode) + "!")
        exit(1)
    
    resource_pool_.set("raccu_reg", raccu_reg)

    return True


# def config_loop(clk_, event_pool_, resource_pool_, handler_pool_, args):
#     raccu_reg = resource_pool_.get("raccu_reg")

#     row = args[0]
#     col = args[1]
#     pc_start = args[2]
#     loop_id = args[3]
#     pc_end = args[4]
#     start_sd = args[5]
#     start = args[6]
#     if start_sd:
#         start = raccu_reg[row][col][start]
#     iter = args[7]
#     step = args[8]

#     if iter <= 0:
#         pc = compute_npc(resource_pool_, row, col, pc_end, 1)
#         resource_pool_.set("pc_" + str(row)+"_"+str(col), pc)
#         return True

#     loop_manager = resource_pool_.get("loop_manager")
#     loop_manager[row][col][loop_id]["enable"] = True
#     loop_manager[row][col][loop_id]["pc_start"] = pc_start+1
#     loop_manager[row][col][loop_id]["pc_end"] = pc_end
#     loop_manager[row][col][loop_id]["iter"] = iter
#     loop_manager[row][col][loop_id]["step"] = step
#     resource_pool_.set("loop_manager", loop_manager)

#     raccu_reg[row][col][15-loop_id] = start
#     resource_pool_.set("raccu_reg", raccu_reg)
#     return True


def control_init(name: str, db: ds.DataBase, prefix: str, event_pool_, resource_pool_, handler_pool_):
    if name == "controller_io" or name == "controller_normal":
        num_drra_row = db.arch.fabric.height
        num_drra_col = db.arch.fabric.width
        for r in range(num_drra_row):
            for c in range(num_drra_col):
                res_name = "status_"+str(r)+"_"+str(c)
                res_value = 0
                resource_pool_.add(res_name, res_value)

        # Loop manager
        resource_pool_.add("loop_manager", [[[{"enable": False, "pc_begin": 0, "pc_end": 0, "start": 0, "step": 0, "iter": 0,
                                               "start_is_dynamic": 0, "step_is_dynamic": 0, "iter_is_dynamic": 0} for i in range(4)] for j in range(num_drra_col)] for k in range(num_drra_row)])
        # RACCU register
        resource_pool_.add(
            "raccu_reg", [[[0 for i in range(16)] for j in range(num_drra_col)] for k in range(num_drra_row)])

        # activation
        handler_pool_.add("resource_trigger", resource_trigger)
        handler_pool_.add("calc_regs", calc_regs)
    else:
        logger.error("Unknown control worker: ", name)
        return False


def control_run(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    ir = resource_pool_.get("ir_"+str(i)+"_"+str(j))

    if ir.name == "wait":
        logger.info("WAIT instruction @ "+str(i)+"_"+str(j))
        exist, mode = get_instr_field(ir, "mode")
        if not exist:
            logger.error("WAIT instruction without mode field")
            return False
        if mode == 0:
            exist, cycle = get_instr_field(ir, "cycle")
            if not exist:
                logger.error("WAIT instruction without cycle field")
                return False
            if cycle > 0:
                pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
                pc = compute_npc(resource_pool_, i, j, pc, 1)
                resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
                e = (clk_+cycle, "fetch_decode_" +
                     str(i)+"_"+str(j), [i, j], 100, True)
                event_pool_.post(e)
            return True
        else:
            logger.debug("Wait for events!")
            exist, cycle = get_instr_field(ir, "cycle")
            if not exist:
                logger.error("Error: Wait instruction without cycle field")
                return False
            status = resource_pool_.get("status_"+str(i)+"_"+str(j))
            if status & cycle == 0:
                pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
                pc = compute_npc(resource_pool_, i, j, pc, 1)
                resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
                e = (clk_+1, "fetch_decode_" +
                     str(i)+"_"+str(j), [i, j], 100, True)
                event_pool_.post(e)
            return True
    elif ir.name == "act":
        logger.info("ACT instruction @ "+str(i)+"_"+str(j))
        exist, ports = get_instr_field(ir, "ports")
        if not exist:
            logger.error("ACT instruction without ports field")
            return False
        exist, mode = get_instr_field(ir, "mode")
        if not exist:
            logger.error("ACT instruction without mode field")
            return False
        exist, param = get_instr_field(ir, "param")
        if not exist:
            logger.error("ACT instruction without param field")
            return False
        if mode == 0:
            for idx_slot in range(4):
                for idx_port in range(4):
                    idx = idx_slot*4+idx_port
                    if ports & (1 << idx) != 0:
                        slot = param + idx_slot
                        port = idx_port
                        e = (clk_+1, "resource_trigger",
                             [i, j, slot, port], 100, True)
                        event_pool_.post(e)
        pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
        pc = compute_npc(resource_pool_, i, j, pc, 1)
        resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
        e = (clk_+1, "fetch_decode_" +
                     str(i)+"_"+str(j), [i, j], 100, True)
        event_pool_.post(e)
        return True
    elif ir.name == "calc":
        logger.info("CALC instruction @ "+str(i)+"_"+str(j))
        exist, mode = get_instr_field(ir, "mode")
        if not exist:
            logger.error("CALC instruction without mode field")
            return False
        exist, operand1_sd = get_instr_field(ir, "operand1_sd")
        if not exist:
            logger.error("CALC instruction without operand1_sd field")
            return False
        exist, operand2_sd = get_instr_field(ir, "operand2_sd")
        if not exist:
            logger.error("CALC instruction without operand2_sd field")
            return False
        exist, operand1 = get_instr_field(ir, "operand1")
        if not exist:
            logger.error("CALC instruction without operand1 field")
            return False
        exist, operand2 = get_instr_field(ir, "operand2")
        if not exist:
            logger.error("CALC instruction without operand2 field")
            return False
        exist, result = get_instr_field(ir, "result")
        if not exist:
            logger.error("CALC instruction without result field")
            return False
        pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
        pc = compute_npc(resource_pool_, i, j, pc, 1)
        resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
        e = (clk_+1, "fetch_decode_" +
             str(i)+"_"+str(j), [i, j], 100, True)
        event_pool_.post(e)
        e = (clk_, "calc_regs", [
             i, j, mode, operand1_sd, operand1, operand2_sd, operand2, result], 100, True)
        event_pool_.post(e)
        return True
    elif ir.name == "looph":
        logger.info("LOOPH instruction @ "+str(i)+"_"+str(j))
        exist, loopid = get_instr_field(ir, "id")
        if not exist:
            logger.error("LOOPH instruction without id field")
            return False
        exist, iter = get_instr_field(ir, "iter")
        if not exist:
            logger.error("LOOPH instruction without iter field")
            return False
        pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
        pc = compute_npc(resource_pool_, i, j, pc, 1)
        resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
        e = (clk_+1, "fetch_decode_" +
             str(i)+"_"+str(j), [i, j], 100, True)
        event_pool_.post(e)
        loop_manager = resource_pool_.get("loop_manager")
        if loop_manager[i][j][loopid]["enable"]:
            logger.error("Error: LOOP instruction without corresponding LOOPT instruction")
            sys.exit(1)
        loop_manager[i][j][loopid]["iter"] = iter
        loop_manager[i][j][loopid]["enable"] = True
        resource_pool_.set("loop_manager", loop_manager)
        return True
    elif ir.name == "loopt":
        logger.info("LOOPT instruction @ "+str(i)+"_"+str(j))
        exist, loopid = get_instr_field(ir, "id")
        if not exist:
            logger.error("LOOPT instruction without id field")
            return False
        exist, loop_pc = get_instr_field(ir, "pc")
        if not exist:
            logger.error("LOOPT instruction without pc field")
            return False

        pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))

        loop_manager = resource_pool_.get("loop_manager")
        if not loop_manager[i][j][loopid]["enable"]:
            logger.error("Error: LOOPT instruction without corresponding LOOP instruction")
            sys.exit(1)
        iter = loop_manager[i][j][loopid]["iter"]
        iter -= 1
        if iter <= 0:
            loop_manager[i][j][loopid]["enable"] = False
            pc = compute_npc(resource_pool_, i, j, pc, 1)
        else:
            loop_manager[i][j][loopid]["iter"] = iter
            pc = compute_npc(resource_pool_, i, j, pc, loop_pc)
        resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
        resource_pool_.set("loop_manager", loop_manager)
        e = (clk_+1, "fetch_decode_" +
             str(i)+"_"+str(j), [i, j], 100, True)
        event_pool_.post(e)
        return True
    return False


def resource_init(name: str, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    db = resource_pool_.get("db")
    slot = args[2]

    resource = None
    for res in db.arch.resources:
        if res.name == name:
            resource = res
            break
    if resource == None:
        logger.error("Error: Unknown resource: ", name)
        return False

    if name == "rf":
        # Register file
        depth = 64 # default depth
        if len(resource.custom_properties)>0:
            for attr in resource.custom_properties:
                if attr.key == "depth":
                    depth = attr.val
                    break

        resource_pool_.add("rf_{}_{}_{}_reg".format(
            i, j, slot), [0 for i in range(depth)])

        storage_map = resource_pool_.get("storage_map")
        resource_map = resource_pool_.get("resource_map")
        conf = resource_pool_.get("conf")
        resource_conf = {}
        for x in range(resource.word_input_port):
            storage_map["rf_{}_{}_{}_{}_conf".format(
                i, j, slot+x, 0)] = "rf_{}_{}_{}_reg".format(i, j, slot)
            conf["rf_{}_{}_{}_{}_conf".format(i, j, slot+x, 0)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
            resource_map["{}_{}_{}".format(i, j, slot+x)] = name
        for x in range(resource.word_output_port):
            storage_map["rf_{}_{}_{}_{}_conf".format(
                i, j, slot+x, 1)] = "rf_{}_{}_{}_reg".format(i, j, slot)
            conf["rf_{}_{}_{}_{}_conf".format(i, j, slot+x, 1)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
            resource_map["{}_{}_{}".format(i, j, slot+x)] = name
        for x in range(resource.bulk_input_port):
            storage_map["rf_{}_{}_{}_{}_conf".format(
                i, j, slot+x, 2)] = "rf_{}_{}_{}_reg".format(i, j, slot)
            conf["rf_{}_{}_{}_{}_conf".format(i, j, slot+x, 2)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
            resource_map["{}_{}_{}".format(i, j, slot+x)] = name
        for x in range(resource.bulk_output_port):
            storage_map["rf_{}_{}_{}_{}_conf".format(
                i, j, slot+x, 3)] = "rf_{}_{}_{}_reg".format(i, j, slot)
            conf["rf_{}_{}_{}_{}_conf".format(i, j, slot+x, 3)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
            resource_map["{}_{}_{}".format(i, j, slot+x)] = name
        
        resource_pool_.set("conf", conf)

        resource_pool_.set("storage_map", storage_map)
        resource_pool_.set("resource_map", resource_map)

        handler_pool_.add("write_word", write_word)
        handler_pool_.add("read_word", read_word)
        handler_pool_.add("write_bulk", write_bulk)
        handler_pool_.add("read_bulk", read_bulk)
        return True
    elif name == "iosram":
        # I/O SRAM
        depth = 64  # default depth
        if len(resource.custom_properties)>0:
            for attr in resource.custom_properties:
                if attr.key == "depth":
                    depth = attr.val
                    break

        resource_pool_.add("iosram_{}_{}_{}_data".format(
            i, j, slot), [[0 for i in range(16)] for j in range(depth)])
        resource_pool_.add("io_temp_var".format(
            i, j, slot), {"input2sram": {}, "sram2output": {}})
        io_temp_var = resource_pool_.get("io_temp_var")
        conf = resource_pool_.get("conf")
        storage_map = resource_pool_.get("storage_map")
        resource_map = resource_pool_.get("resource_map")
        for x in range(resource.bulk_input_port):
            storage_map["iosram_{}_{}_{}_{}_conf".format(
                i, j, slot+1+x, 2)] = "iosram_{}_{}_{}_data".format(i, j, slot)
            conf["iosram_{}_{}_{}_{}_conf".format(i, j, slot+x, 2)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
            resource_map["{}_{}_{}".format(i, j, slot+x)] = name
        for x in range(resource.bulk_output_port):
            storage_map["iosram_{}_{}_{}_{}_conf".format(
                i, j, slot+1+x, 3)] = "iosram_{}_{}_{}_data".format(i, j, slot)
            conf["iosram_{}_{}_{}_{}_conf".format(i, j, slot+x, 3)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
            resource_map["{}_{}_{}".format(i, j, slot+x)] = name

        # define port for I/O interface and the paired bulk ports
        resource_map["{}_{}_{}".format(i, j, slot)] = name
        storage_map["iosram_{}_{}_{}_{}_conf".format(
                i, j, slot, 0)] = "iosram_{}_{}_{}_data".format(i, j, slot)
        conf["iosram_{}_{}_{}_{}_conf".format(i, j, slot, 0)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        storage_map["iosram_{}_{}_{}_{}_conf".format(
                i, j, slot, 1)] = "iosram_{}_{}_{}_data".format(i, j, slot)
        conf["iosram_{}_{}_{}_{}_conf".format(i, j, slot, 1)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        storage_map["iosram_{}_{}_{}_{}_conf".format(
                i, j, slot, 2)] = "iosram_{}_{}_{}_data".format(i, j, slot)
        conf["iosram_{}_{}_{}_{}_conf".format(i, j, slot, 2)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        storage_map["iosram_{}_{}_{}_{}_conf".format(
                i, j, slot, 3)] = "iosram_{}_{}_{}_data".format(i, j, slot)
        conf["iosram_{}_{}_{}_{}_conf".format(i, j, slot, 3)] = {"start": 0, "repeat": [
                {"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        io_temp_var["input2sram"]["iosram_{}_{}_{}".format(i, j, slot)]= 0
        io_temp_var["sram2output"]["iosram_{}_{}_{}".format(i, j, slot)]= 0
        

        resource_pool_.set("storage_map", storage_map)
        resource_pool_.set("resource_map", resource_map)
        resource_pool_.set("conf", conf)

        handler_pool_.add("write_bulk", write_bulk)
        handler_pool_.add("read_bulk", read_bulk)
        handler_pool_.add("io_write_to_sram", io_write_to_sram)
        handler_pool_.add("sram_read_to_io", sram_read_to_io)
        handler_pool_.add("read_from_io", read_from_io)
        handler_pool_.add("write_to_io", write_to_io)
        return True
    elif name == "dpu":
        dpu_internal_regs = resource_pool_.get("dpu_internal_regs")
        dpu_internal_regs["{}_{}_{}".format(i, j, slot)] = {
            "acc": 0, "scalar": 0}
        resource_pool_.set("dpu_internal_regs", dpu_internal_regs)
        resource_map = resource_pool_.get("resource_map")
        conf = resource_pool_.get("conf")
        conf["dpu_{}_{}_{}_conf".format(i, j, slot)] = {
            "option": [{"mode": 0, "imm": 0} for i in range(4)], "delay": [0 for i in range(3)], "max_state": 0, "repeat": [{"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        resource_pool_.set("conf", conf)
        resource_map["{}_{}_{}".format(i, j, slot)] = name
        resource_pool_.set("resource_map", resource_map)
        handler_pool_.add("set_curr_dpu_mode", set_current_dpu_mode)
        return True
    elif name == "swb":
        connection_map = resource_pool_.get("connection_map")
        conn = {"word": {}, "bulk": {}}
        connection_map["{}_{}_{}".format(i, j, slot)] = conn
        resource_pool_.set("connection_map", connection_map)
        resource_map = resource_pool_.get("resource_map")
        conf = resource_pool_.get("conf")
        conf["swb_{}_{}_{}_{}_conf".format(i, j, slot, 0)] = {
            "option": [{} for i in range(4)], "delay": [0 for i in range(3)], "max_state": 0, "repeat": [{"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        conf["swb_{}_{}_{}_{}_conf".format(i, j, slot, 2)] = {
            "option": [{} for i in range(4)], "delay": [0 for i in range(3)], "max_state": 0, "repeat": [{"iter": 1, "step": 0, "delay": 0} for i in range(8)]}
        resource_pool_.set("conf", conf)
        resource_map["{}_{}_{}".format(i, j, slot)] = name
        resource_pool_.set("resource_map", resource_map)
        handler_pool_.add("set_curr_swb_mode", set_current_swb_mode)
        handler_pool_.add("set_curr_route_mode", set_current_route_mode)
        return True
    else:
        logger.error("Error: Unknown resource worker: ")
        return False
    return False


def resource_run(clk_, event_pool_, resource_pool_, handler_pool_, args):
    i = args[0]
    j = args[1]
    db = resource_pool_.get("db")
    ir = resource_pool_.get("ir_"+str(i)+"_"+str(j))
    exist, slot = get_instr_field(ir, "slot")
    if not exist:
        logger.error("Error: Instruction without slot field")
        return False
    prefix = str(i)+"_"+str(j)+"_"+str(slot)
    resource_map = resource_pool_.get("resource_map")
    resource_name = resource_map[prefix]
    logger.debug(resource_name)

    if resource_name == "rf":
        if ir.name == "dsu":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf[
                "{}_{}_{}_{}_{}_conf".format(resource_name, i, j, slot, port)]
            exist, init_addr_sd = get_instr_field(ir, "init_addr_sd")
            if not exist:
                logger.error("Error: Instruction without start_sd field")
                return False
            exist, resource_conf["start"] = get_instr_field(ir, "init_addr")
            if not exist:
                logger.error("Error: Instruction without init_addr field")
                return False
            if init_addr_sd != 0:
                raccu_reg = resource_pool_.get("raccu_reg")
                resource_conf["start"] = raccu_reg[i][j][resource_conf["start"]]

            resource_pool_.set("conf", conf)
            logger.info("DSU instr: start="+str(resource_conf["start"]))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        if ir.name == "rep":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["{}_{}_{}_{}_{}_conf".format(resource_name, i, j, slot, port)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 0)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 0)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 0)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level="+str(level) +
                        ", iter="+str(iter)+", step="+str(step)+", delay="+str(delay))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == "repx":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["{}_{}_{}_{}_{}_conf".format(
                resource_name, i, j, slot, port)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 6)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 6)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 6)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level={}, iter={}, step={}, delay={}".format(
                level, resource_conf["repeat"][level]["iter"], resource_conf["repeat"][level]["step"], resource_conf["repeat"][level]["delay"]))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
    elif resource_name == "iosram":
        if ir.name == "dsu":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["iosram_{}_{}_{}_{}_conf".format(
                i, j, slot, port)]
            exist, init_addr_sd = get_instr_field(ir, "init_addr_sd")
            if not exist:
                logger.error("Error: Instruction without init_addr_sd field")
                return False
            exist, resource_conf["start"] = get_instr_field(ir, "init_addr")
            if not exist:
                logger.error("Error: Instruction without init_addr field")
                return False
            if init_addr_sd != 0:
                raccu_reg = resource_pool_.get("raccu_reg")
                resource_conf["start"] = raccu_reg[i][j][resource_conf["start"]]
            resource_pool_.set("conf", conf)
            logger.info("DSU instr: start="+str(resource_conf["start"]))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        # elif ir.name == "io":
        #     exist, port = get_instr_field(ir, "port")
        #     if not exist:
        #         logger.error("Error: Instruction without port field")
        #         return False
        #     conf = resource_pool_.get("conf")
        #     resource_conf = conf["iosram_{}_{}_{}_{}_conf".format(
        #         i, j, slot, port)]
        #     exist, ext_addr_sd = get_instr_field(ir, "ext_addr_sd")
        #     if not exist:
        #         logger.error("Error: Instruction without ext_addr_sd field")
        #         return False
        #     exist, resource_conf["ext_addr"] = get_instr_field(ir, "ext_addr")
        #     if not exist:
        #         logger.error("Error: Instruction without ext_addr field")
        #         return False
        #     if ext_addr_sd != 0:
        #         raccu_reg = resource_pool_.get("raccu_reg")
        #         resource_conf["ext_addr"] = raccu_reg[i][j][resource_conf["ext_addr"]]
        #     exist, resource_conf["int_addr"] = get_instr_field(ir, "int_addr")
        #     if not exist:
        #         logger.error("Error: Instruction without int_addr field")
        #         return False
        #     resource_pool_.set("conf", conf)
        #     logger.info("IO instr: ext_addr="+str(resource_conf["ext_addr"]) +
        #                 ", int_addr="+str(resource_conf["int_addr"]))
        #     pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
        #     pc = compute_npc(resource_pool_, i, j, pc, 1)
        #     resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
        #     e = (clk_+1, "fetch_decode_" +
        #          str(i)+"_"+str(j), [i, j], 100, True)
        #     event_pool_.post(e)
        #     return True
        elif ir.name == "rep":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["{}_{}_{}_{}_{}_conf".format(
                resource_name, i, j, slot, port)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 0)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 0)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 0)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level="+str(level) +
                        ", iter="+str(iter)+", step="+str(step)+", delay="+str(delay))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == "repx":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["{}_{}_{}_{}_{}_conf".format(
                resource_name, i, j, slot, port)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 6)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 6)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 6)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level={}, iter={}, step={}, delay={}".format(
                level, resource_conf["repeat"][level]["iter"], resource_conf["repeat"][level]["step"], resource_conf["repeat"][level]["delay"]))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
    elif resource_name == 'dpu':
        if ir.name == 'dpu':
            conf = resource_pool_.get("conf")
            resource_conf = conf["dpu_{}_{}_{}_conf".format(i, j, slot)]
            exist, mode = get_instr_field(ir, "mode")
            if not exist:
                logger.error("Error: Instruction without mode field")
                return False
            exist, immediate = get_instr_field(ir, "immediate")
            if not exist:
                logger.error("Error: Instruction without immediate field")
                return False
            exist, option = get_instr_field(ir, "option")
            if not exist:
                logger.error("Error: Instruction without option field")
                return False
            resource_conf["option"][option]["mode"] = mode
            resource_conf["option"][option]["imm"] = immediate
            resource_pool_.set("conf", conf)
            logger.info("DPU instr: mode="+str(mode) +
                        ", immediate="+str(immediate) + " in option "+str(option))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == 'rep':
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["dpu_{}_{}_{}_conf".format(i, j, slot)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 0)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 0)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 0)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level="+str(level) +
                        ", iter="+str(iter)+", step="+str(step)+", delay="+str(delay))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == "repx":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["{}_{}_{}_{}_conf".format(
                resource_name, i, j, slot)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 6)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 6)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 6)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level={}, iter={}, step={}, delay={}".format(
                level, resource_conf["repeat"][level]["iter"], resource_conf["repeat"][level]["step"], resource_conf["repeat"][level]["delay"]))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == 'fsm':
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, delay_0 = get_instr_field(ir, "delay_0")
            if not exist:
                logger.error("Error: Instruction without delay_0 field")
                return False
            exist, delay_1 = get_instr_field(ir, "delay_1")
            if not exist:
                logger.error("Error: Instruction without delay_1 field")
                return False
            exist, delay_2 = get_instr_field(ir, "delay_2")
            if not exist:
                logger.error("Error: Instruction without delay_2 field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["dpu_{}_{}_{}_conf".format(
                i, j, slot)]
            resource_conf["delay"][0] = delay_0
            resource_conf["delay"][1] = delay_1
            resource_conf["delay"][2] = delay_2
            resource_pool_.set("conf", conf)
            logger.info("FSM instr: delay_0="+str(delay_0) + ", delay_1="+str(delay_1) +
                        ", delay_2="+str(delay_2))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
    if resource_name == 'swb':
        if ir.name == 'swb':
            conf = resource_pool_.get("conf")
            resource_conf = conf["swb_{}_{}_{}_{}_conf".format(i, j, slot, 0)]
            exist, source = get_instr_field(ir, "source")
            if not exist:
                logger.error("Error: Instruction without source field")
                return False
            exist, target = get_instr_field(ir, "target")
            if not exist:
                logger.error("Error: Instruction without target field")
                return False
            exist, option = get_instr_field(ir, "option")
            if not exist:
                logger.error("Error: Instruction without option field")
                return False

            if resource_conf["max_state"] < option:
                resource_conf["max_state"] = option

            resource_conf["option"][option]["{}_{}_{}_{}".format(
                i, j, target, 0)] = "{}_{}_{}_{}".format(
                i, j, source, 1)
            resource_pool_.set("conf", conf)
            logger.info("SWB instr: source="+str(source) +
                        ", target="+str(target) + " in option "+str(option))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == 'route':
            conf = resource_pool_.get("conf")
            resource_conf = conf["swb_{}_{}_{}_{}_conf".format(i, j, slot, 2)]
            exist, sr = get_instr_field(ir, "sr")
            if not exist:
                logger.error("Error: Instruction without sr field")
                return False
            exist, source = get_instr_field(ir, "source")
            if not exist:
                logger.error("Error: Instruction without source field")
                return False
            exist, target = get_instr_field(ir, "target")
            if not exist:
                logger.error("Error: Instruction without target field")
                return False
            exist, option = get_instr_field(ir, "option")
            if not exist:
                logger.error("Error: Instruction without option field")
                return False

            if resource_conf["max_state"] < option:
                resource_conf["max_state"] = option

            if sr == 0:
                # send
                source_str = "{}_{}_{}_{}".format(i, j, source, 3)
                if target & 0x01 != 0:
                    # north-west
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i-1, j-1)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x02 != 0:
                    # north
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i-1, j)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x04 != 0:
                    # north-east
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i-1, j+1)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x08 != 0:
                    # west
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i, j-1)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x10 != 0:
                    # center
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i, j)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x20 != 0:
                    # east
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i, j+1)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x40 != 0:
                    # south-west
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i+1, j-1)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x80 != 0:
                    # south
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i+1, j)
                    resource_conf["option"][option][target_str] = source_str
                if target & 0x100 != 0:
                    # south-east
                    target_str = "temp_{}_{}_to_{}_{}".format(i, j, i+1, j+1)
                    resource_conf["option"][option][target_str] = source_str
            else:
                # receive
                target_str_array = []
                for idx in range(16):
                    if target & (1 << idx) != 0:
                        target_str_array.append("{}_{}_{}_{}".format(
                            i, j, idx, 2))
                if source == 0:
                    # north-west
                    source_str = "temp_{}_{}_to_{}_{}".format(i-1, j-1, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 1:
                    # north
                    source_str = "temp_{}_{}_to_{}_{}".format(i-1, j, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 2:
                    # north-east
                    source_str = "temp_{}_{}_to_{}_{}".format(i-1, j+1, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 3:
                    # west
                    source_str = "temp_{}_{}_to_{}_{}".format(i, j-1, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 4:
                    # center
                    source_str = "temp_{}_{}_to_{}_{}".format(i, j, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 5:
                    # east
                    source_str = "temp_{}_{}_to_{}_{}".format(i, j+1, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 6:
                    # south-west
                    source_str = "temp_{}_{}_to_{}_{}".format(i+1, j-1, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 7:
                    # south
                    source_str = "temp_{}_{}_to_{}_{}".format(i+1, j, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
                elif source == 8:
                    # south-east
                    source_str = "temp_{}_{}_to_{}_{}".format(i+1, j+1, i, j)
                    for target_str in target_str_array:
                        resource_conf["option"][option][target_str] = source_str
            resource_pool_.set("conf", conf)
            logger.info("ROUTE instr: sr="+str(sr) +
                        ", source="+str(source)+", target="+str(target) + " in option "+str(option))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == 'rep':
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["swb_{}_{}_{}_{}_conf".format(
                i, j, slot, port)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 0)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 0)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 0)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level="+str(level) +
                        ", iter="+str(iter)+", step="+str(step)+", delay="+str(delay))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == "repx":
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, level = get_instr_field(ir, "level")
            if not exist:
                logger.error("Error: Instruction without level field")
                return False
            exist, iter = get_instr_field(ir, "iter")
            if not exist:
                logger.error("Error: Instruction without iter field")
                return False
            exist, step = get_instr_field(ir, "step")
            if not exist:
                logger.error("Error: Instruction without step field")
                return False
            exist, delay = get_instr_field(ir, "delay")
            if not exist:
                logger.error("Error: Instruction without delay field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["{}_{}_{}_{}_{}_conf".format(
                resource_name, i, j, slot, port)]
            resource_conf["repeat"][level]["iter"] = partial_update_variable(resource_conf["repeat"][level]["iter"], iter, 6, 6)
            resource_conf["repeat"][level]["step"] = partial_update_variable(resource_conf["repeat"][level]["step"], step, 6, 6)
            resource_conf["repeat"][level]["delay"] = partial_update_variable(resource_conf["repeat"][level]["delay"], delay, 6, 6)
            resource_pool_.set("conf", conf)
            logger.info("REP instr: level={}, iter={}, step={}, delay={}".format(
                level, resource_conf["repeat"][level]["iter"], resource_conf["repeat"][level]["step"], resource_conf["repeat"][level]["delay"]))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
        elif ir.name == 'fsm':
            exist, port = get_instr_field(ir, "port")
            if not exist:
                logger.error("Error: Instruction without port field")
                return False
            exist, delay_0 = get_instr_field(ir, "delay_0")
            if not exist:
                logger.error("Error: Instruction without delay_0 field")
                return False
            exist, delay_1 = get_instr_field(ir, "delay_1")
            if not exist:
                logger.error("Error: Instruction without delay_1 field")
                return False
            exist, delay_2 = get_instr_field(ir, "delay_2")
            if not exist:
                logger.error("Error: Instruction without delay_2 field")
                return False
            conf = resource_pool_.get("conf")
            resource_conf = conf["swb_{}_{}_{}_{}_conf".format(
                i, j, slot, port)]
            resource_conf["delay"][0] = delay_0
            resource_conf["delay"][1] = delay_1
            resource_conf["delay"][2] = delay_2
            resource_pool_.set("conf", conf)
            logger.info("FSM instr: delay_0="+str(delay_0) + ", delay_1="+str(delay_1) +
                        ", delay_2="+str(delay_2))
            pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
            pc = compute_npc(resource_pool_, i, j, pc, 1)
            resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
            e = (clk_+1, "fetch_decode_" +
                 str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            return True
    return False


def propagate_connection(curr_value, connection_map):
    bulk_conn = {}
    for cell in connection_map:
        for dest in connection_map[cell]["bulk"]:
            bulk_conn[dest] = connection_map[cell]["bulk"][dest]
    for cell in connection_map:
        row = int(cell.split("_")[0])
        col = int(cell.split("_")[1])
        slot = int(cell.split("_")[2])
        word_conn = connection_map[cell]["word"]
        for dest in word_conn:
            dest_row = int(dest.split("_")[0])
            dest_col = int(dest.split("_")[1])
            dest_slot = int(dest.split("_")[2])
            dest_port = int(dest.split("_")[3])
            src = word_conn[dest]
            src_row = int(src.split("_")[0])
            src_col = int(src.split("_")[1])
            src_slot = int(src.split("_")[2])
            src_port = int(src.split("_")[3])
            value = curr_value[src_row][src_col][src_slot][src_port]
            curr_value[dest_row][dest_col][dest_slot][dest_port] = value
        for dest in bulk_conn:
            if dest.startswith("temp"):
                continue
            dest_row = int(dest.split("_")[0])
            dest_col = int(dest.split("_")[1])
            dest_slot = int(dest.split("_")[2])
            dest_port = int(dest.split("_")[3])
            src = bulk_conn[dest]
            if src.startswith("temp"):
                for x in bulk_conn:
                    if x == src:
                        src = bulk_conn[x]
                        break
            if src.startswith("temp"):
                continue
            src_row = int(src.split("_")[0])
            src_col = int(src.split("_")[1])
            src_slot = int(src.split("_")[2])
            src_port = int(src.split("_")[3])
            value = curr_value[src_row][src_col][src_slot][src_port]
            curr_value[dest_row][dest_col][dest_slot][dest_port] = value
    return curr_value


def comb_callback(clk_, event_pool_, resource_pool_, handler_pool_, args):
    db = resource_pool_.get("db")

    # let the next_value be the curr_value
    #next_value = resource_pool_.get("next_value")
    #curr_value = copy.deepcopy(next_value)
    curr_value = resource_pool_.get("curr_value")

    # propagate the value according to the connection map, DPUs haven't processed the data yet
    connection_map = resource_pool_.get("connection_map")
    curr_value = propagate_connection(curr_value, connection_map)

    # DPUs start to process the data
    resource_map = resource_pool_.get("resource_map")
    curr_dpu_mode = resource_pool_.get("curr_dpu_mode")
    dpu_internal_regs = resource_pool_.get("dpu_internal_regs")
    for r in range(db.arch.fabric.height):
        for c in range(db.arch.fabric.width):
            for s in range(16):
                if "{}_{}_{}".format(r, c, s) not in resource_map:
                    continue
                resource_name = resource_map[str(r)+"_"+str(c)+"_"+str(s)]
                if resource_name == 'dpu':
                    if "{}_{}_{}".format(r, c, s) not in curr_dpu_mode:
                        continue
                    conf = curr_dpu_mode["{}_{}_{}".format(r, c, s)]
                    if conf["active"]:
                        # reset internal registers
                        dpu_internal_regs["{}_{}_{}".format(
                            r, c, s)]["acc"] = 0
                        curr_dpu_mode["{}_{}_{}".format(r, c, s)]['active']=False
                        resource_pool_.set("curr_dpu_mode", curr_dpu_mode)
                    in0 = curr_value[r][c][s][0]
                    in1 = curr_value[r][c][s+1][0]
                    out = 0
                    acc = dpu_internal_regs["{}_{}_{}".format(r, c, s)]["acc"]
                    scalar = dpu_internal_regs["{}_{}_{}".format(
                        r, c, s)]["scalar"]
                    imm = conf["imm"]
                    if conf["mode"] == 0:
                        out = 0
                    elif conf["mode"] == 1:
                        out = in0 + in1
                    elif conf["mode"] == 2:
                        print("old_acc= %d" % (acc))
                        acc = in0*in1 + acc
                        out = acc
                        print("in0= %d, in1= %d, acc= %d, out=%d" % (in0, in1, acc, out))
                    elif conf["mode"] == 3:
                        out = max(in0, in1)
                    elif conf["mode"] == 7:
                        out = in0*in1
                    else:
                        logger.error("Error: Unknown DPU mode")
                        return False
                    dpu_internal_regs["{}_{}_{}".format(r, c, s)]["acc"] = acc
                    dpu_internal_regs["{}_{}_{}".format(
                        r, c, s)]["scalar"] = scalar

                    print("out= " + str(out))
                    e = [clk_+1, "delay_signal", [r, c, s, 1, out], 100, False]
                    event_pool_.post(e)
    resource_pool_.set("dpu_internal_regs", dpu_internal_regs)

    # set curr_value
    resource_pool_.set("curr_value", curr_value)

    # clear all IO related temp var
    io_temp_var = resource_pool_.get("io_temp_var")
    for tag in io_temp_var["input2sram"]:
        io_temp_var["input2sram"][tag] = 0
    for tag in io_temp_var["sram2output"]:
        io_temp_var["sram2output"][tag] = 0

    # Post a comb event again
    e = (clk_+1, "comb_callback", [], 50, False)
    event_pool_.post(e)

    return True
