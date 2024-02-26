import re
import os
import logging
import verboselogs
import sys
import json
import numpy as np
import Parser as par
import DataStructure_pb2 as ds
import WorkerManager as wm

logger = verboselogs.VerboseLogger('vesim')


class ShutdownHandler(logging.Handler):
    def emit(self, record):
        logging.error(record.msg)
        logging.shutdown()
        sys.exit(1)


logging.getLogger().addHandler(ShutdownHandler(level=50))


def get_cell_name(db: ds.DataBase, coord: ds.Coordinate) -> str:
    for cell in db.arch.fabric.cell_lists:
        if coord in cell.coordinates:
            return cell.cell_name
    return None


def get_controller_name(db: ds.DataBase, cell_name: str) -> str:
    for cell in db.arch.cells:
        if cell.name == cell_name:
            return cell.controller
    return None


def get_resource_list(db: ds.DataBase, cell_name: str) -> str:
    for cell in db.arch.cells:
        if cell.name == cell_name:
            return cell.resource_list
    return None


def fetch_decode(clk_, event_pool_, resource_pool_, handler_pool_, args):
    db = resource_pool_.get("db")
    pc = resource_pool_.get("pc_"+str(args[0])+"_"+str(args[1]))
    label = str(args[0])+"_"+str(args[1])
    instr_list = None
    for il in db.pkg.instruction_lists:
        if il.label == label:
            instr_list = il
            break
    if instr_list == None:
        logger.fatal("Error: Instruction list not found: ", label)
        return False

    instr = instr_list.instructions[pc]
    resource_pool_.set("ir_"+str(args[0])+"_"+str(args[1]), instr)
    if wm.is_resource_instr(instr.name, db.isa):
        # resource instruction
        if not wm.resource_run(clk_, event_pool_, resource_pool_, handler_pool_, args):
            logger.fatal("Failed to run resource instruction: " + instr.name)
            return False
    else:
        # non-resource instruction
        if not wm.control_run(clk_, event_pool_, resource_pool_, handler_pool_, args):
            logger.fatal(
                "Failed to run control instruction: " + instr.name)
            return False

    return True


def reset_interconnection_value(clk_, event_pool_, resource_pool_, handler_pool_, args):
    num_drra_row = args[0]
    num_drra_col = args[1]

    curr_value = [[[[0 for l in range(4)] for k in range(
        16)] for j in range(num_drra_col)] for i in range(num_drra_row)]
    resource_pool_.set("curr_value", curr_value)

    e = (clk_+1, "reset_interconnection_value", [
         num_drra_row, num_drra_col], 1, False)
    event_pool_.post(e)
    return True


def init_event(event_pool_, resource_pool_, handler_pool_, file_arch_, file_isa_, file_instr_, file_input_, file_output_, file_state_reg_):

    # Load Instruction, architecture and instruction set
    db = ds.DataBase()
    parser = par.Parser()
    if not parser.load_isa(db.isa, file_isa_):
        logger.fatal("Failed to load instruction set file: " + file_isa_)
        exit(-1)
    if not parser.load_arch(db.arch, file_arch_):
        logger.fatal("Failed to load architecture file: " + file_arch_)
        exit(-1)
    if not parser.load_instr(db.arch, db.pkg, db.isa, file_instr_):
        logger.fatal("Failed to load instruction file: " + file_instr_)
        exit(-1)

    # get arch parameters
    num_drra_row = db.arch.fabric.height
    num_drra_col = db.arch.fabric.width
    input_buffer_depth = db.arch.interface.input_buffer_depth
    output_buffer_depth = db.arch.interface.output_buffer_depth

    resource_pool_.add("db", db)

    # Resource to record IO activity
    resource_pool_.add("IAP", [])
    resource_pool_.add("OAP", [])

    # Load Input, Output buffer and scratch pad memory     
    resource_pool_.add("input_buffer", {})
    resource_pool_.add("output_buffer", {})
    resource_pool_.add("output_buffer_active", [False
                                                for j in range(output_buffer_depth)])
    
    input_buffer = resource_pool_.get("input_buffer")
    with open(file_input_, "r") as f:
        content = f.read().split("\n")
        for line in content:
            result = re.match(r"(\d+)\s+([01]+)", line)
            if result:
                addr = int(result.group(1))
                data = [0] * 16
                for i in range(16):
                    data[16-i-1] = int(result.group(2)[i*16:(i+1)*16], 2)
                input_buffer[addr] = data
    resource_pool_.set("input_buffer", input_buffer)
    
    if os.path.exists(file_output_):
        output_buffer = resource_pool_.get("output_buffer")
        output_buffer_active = resource_pool_.get("output_buffer_active")
        with open(file_output_, "r") as f:
            content = f.read().split("\n")
            for line in content:
                result = re.match(r"(\d+)\s+([01]+)", line)
                if result:
                    addr = int(result.group(1))
                    data = [0] * 16
                    for i in range(16):
                        data[16-i-1] = int(result.group(2)[i*16:(i+1)*16], 2)
                    output_buffer[addr] = data
                    output_buffer_active[addr] = True
        resource_pool_.set("output_buffer", output_buffer)
        resource_pool_.set("output_buffer_active", output_buffer_active)
    
    #print output buffer
    output_buffer = resource_pool_.get("output_buffer")
    output_buffer_active = resource_pool_.get("output_buffer_active")

    # add handler
    handler_pool_.add("delay_signal", wm.delay_signal)

    # add storage map and resource map
    resource_pool_.add("storage_map", {})
    resource_pool_.add("resource_map", {})
    resource_pool_.add("conf", {})
    resource_pool_.add("curr_dpu_mode", {})
    resource_pool_.add("dpu_internal_regs", {})
    # Add connection map
    resource_pool_.add("connection_map", {})
    # Add port variables
    resource_pool_.add("curr_value", [[[[0 for l in range(4)] for k in range(
        16)] for j in range(num_drra_col)] for i in range(num_drra_row)])
    resource_pool_.add("next_value", [[[[0 for l in range(4)] for k in range(
        16)] for j in range(num_drra_col)] for i in range(num_drra_row)])

    # Init control and resource workers according to arch.json
    for r in range(num_drra_row):
        for c in range(num_drra_col):
            coord = ds.Coordinate()
            coord.row = r
            coord.col = c
            cell_name = get_cell_name(db, coord)
            if not cell_name:
                logger.fatal("Cell not found: " + str(coord))
                exit(-1)
            controller_name = get_controller_name(db, cell_name)
            if not controller_name:
                logger.fatal("Controller not found: " + cell_name)
                exit(-1)
            wm.control_init(controller_name, db, str(r)+"_"+str(c), event_pool_,
                            resource_pool_, handler_pool_)
            resource_list = get_resource_list(db, cell_name)
            current_slot = 0
            for resource_name in resource_list:
                wm.resource_init(resource_name, event_pool_,
                                 resource_pool_, handler_pool_, [r, c, current_slot])
                resource = None
                for rs in db.arch.resources:
                    if rs.name == resource_name:
                        resource = rs
                        break
                if not resource:
                    logger.error("Resource not found: " + resource_name)
                current_slot = current_slot + resource.size

    # read file contents as dictionary
    if os.path.exists(file_state_reg_):
        with open(file_state_reg_, "r") as f:
            raccu_reg =resource_pool_.get("raccu_reg")
            state_reg = json.load(f)
            for label in state_reg:
                [row, col, addr] = label.split("_")
                raccu_reg[int(row)][int(col)][int(addr)] = state_reg[label]
            resource_pool_.set("raccu_reg", raccu_reg)

    resource_map = resource_pool_.get("resource_map")
    print(resource_map)

    # Create IR and post event for fetch
    for i in range(num_drra_row):
        for j in range(num_drra_col):
            resource_pool_.add("ir_"+str(i)+"_"+str(j), None)
            resource_pool_.add("pc_"+str(i)+"_"+str(j), 0)
            e = (0+j, "fetch_decode_"+str(i)+"_"+str(j), [i, j], 100, True)
            event_pool_.post(e)
            handler_pool_.add("fetch_decode_"+str(i) +
                              "_"+str(j), fetch_decode)

    # Register for combinatorial event
    handler_pool_.add("comb_callback", wm.comb_callback)
    e = (0, "comb_callback", [], 50, False)
    event_pool_.post(e)

    # post event for reset interconnection value
    handler_pool_.add("reset_interconnection_value",
                      reset_interconnection_value)
    e = (0, "reset_interconnection_value", [
         num_drra_row, num_drra_col], 1, False)
    event_pool_.post(e)

    #     resource_pool_.add("regfile", [[[0 for i in range(64)] for j in range(
    #         num_drra_col)] for k in range(num_drra_row)])
    #     # Add RF and DPU inputs and outputs
    #     for i in range(num_drra_row):
    #         for j in range(num_drra_col):
    #             dest = str(i)+"_"+str(j)+"_RF_in0"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_RF_in1"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_in0"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_in1"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_in2"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_in3"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_RF_out0"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_RF_out1"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_out0"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_out1"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_out0_reg"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_out1_reg"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_acc0"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_DPU_acc1"
    #             resource_pool_.add(dest, 0)
    #             dest = str(i)+"_"+str(j)+"_RF_in0_dimarch"
    #             resource_pool_.add(dest, [0]*16)
    #             dest = str(i)+"_"+str(j)+"_RF_out0_dimarch"
    #             resource_pool_.add(dest, [0]*16)
    #             dest = str(i)+"_"+str(j)+"_SRAM_in"
    #             resource_pool_.add(dest, [0]*16)
    #             dest = str(i)+"_"+str(j)+"_SRAM_out"
    #             resource_pool_.add(dest, [0]*16)
    #             dest = str(i)+"_"+str(j)+"_IO_in"
    #             resource_pool_.add(dest, [0]*16)
    #             dest = str(i)+"_"+str(j)+"_IO_out"
    #             resource_pool_.add(dest, [0]*16)

    #     # Add DPU internal registers
    #     dpu_internal_regs = [[[0 for i in range(2)] for j in range(
    #         num_drra_col)] for k in range(num_drra_row)]
    #     resource_pool_.add('dpu_internal_regs', dpu_internal_regs)

    #     # Register config REFI
    #     handler_pool_.add("config_refi", self.config_refi)
    #     handler_pool_.add("config_dpu", self.config_dpu)
    #     handler_pool_.add("config_swb", self.config_swb)
    #     handler_pool_.add("config_sram", self.config_sram)
    #     handler_pool_.add("config_io", self.config_io)
    #     handler_pool_.add("rf_read", self.rf_read)
    #     handler_pool_.add("rf_write", self.rf_write)
    #     handler_pool_.add("sram_read", self.sram_read)
    #     handler_pool_.add("sram_write", self.sram_write)
    #     handler_pool_.add("io_read", self.io_read)
    #     handler_pool_.add("io_write", self.io_write)
    #     handler_pool_.add("config_loop", self.config_loop)
    #     handler_pool_.add("config_raccu", self.config_raccu)
    #     handler_pool_.add("config_perm", self.config_perm)
    #     handler_pool_.add("post_dpu_config", self.post_dpu_config)
    #     handler_pool_.add("comb_callback", self.comb_callback)

    #     # Register shadow registers
    #     for i in range(num_drra_row):
    #         for j in range(num_drra_col):
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_REFI_in0", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_REFI_in1", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_REFI_out0", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_REFI_out1", "")
    #             resource_pool_.add(str(i)+"_"+str(j) +
    #                                "_shadow_REFI_in0_dimarch", "")
    #             resource_pool_.add(str(i)+"_"+str(j) +
    #                                "_shadow_REFI_out0_dimarch", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_DPU", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_SWB", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_SRAM", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_RACCU", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_LOOP", "")
    #             resource_pool_.add(str(i)+"_"+str(j)+"_shadow_LOOP_pc", -1)
    #     # Register config
    #     for i in range(num_drra_row):
    #         for j in range(num_drra_col):
    #             resource_pool_.add("dpu_"+str(i)+"_"+str(j)+"_config", {
    #                                "mode": 0, "saturation": False, "fixed": False, "acc_clear": 0, "negate_in0": False, "negate_in1": False, "absolute_out": False, "immediate": False})
    #             resource_pool_.add("refi_"+str(i)+"_"+str(j)+"_r0_config", "")
    #             resource_pool_.add("refi_"+str(i)+"_"+str(j)+"_r1_config", "")
    #             resource_pool_.add("refi_"+str(i)+"_"+str(j)+"_w0_config", "")
    #             resource_pool_.add("refi_"+str(i)+"_"+str(j)+"_w1_config", "")
    #             resource_pool_.add("refi_"+str(i)+"_" +
    #                                str(j)+"_r0_dimarch_config", "")
    #             resource_pool_.add("refi_"+str(i)+"_" +
    #                                str(j)+"_w0_dimarch_config", "")

    #     # Loop manager
    #     resource_pool_.add("loop_manager", [[[{"enable": False, "pc_begin": 0, "pc_end": 0, "start": 0, "step": 0, "iter": 0,
    #                                            "start_is_dynamic": 0, "step_is_dynamic": 0, "iter_is_dynamic": 0} for i in range(4)] for j in range(num_drra_col)] for k in range(num_drra_row)])
    #     # RACCU register
    #     resource_pool_.add(
    #         "raccu_reg", [[[0 for i in range(16)] for j in range(num_drra_col)] for k in range(num_drra_row)])

    #     # post combination events
    #     e = (0, "comb_callback", [], 50, False)
    #     event_pool_.post(e)

    # def compute_npc(self, resource_pool_, row_, col_, pc_, pc_increment_):
    #     loop_manager = resource_pool_.get("loop_manager")
    #     for i in range(4):
    #         curr_loop_id = 3-i
    #         loop_config = loop_manager[row_][col_][curr_loop_id]
    #         if loop_config["enable"]:
    #             if pc_ == loop_config["pc_end"]:
    #                 if loop_config["iter"] <= 1:
    #                     loop_config["enable"] = False
    #                     loop_manager[row_][col_][curr_loop_id] = loop_config
    #                     resource_pool_.set("loop_manager", loop_manager)
    #                     continue
    #                 else:
    #                     loop_config["iter"] = loop_config["iter"] - 1
    #                     raccu_reg = resource_pool_.get("raccu_reg")
    #                     raccu_reg[row_][col_][15 -
    #                                           curr_loop_id] += loop_config["step"]
    #                     resource_pool_.set("raccu_reg", raccu_reg)
    #                     loop_manager[row_][col_][curr_loop_id] = loop_config
    #                     resource_pool_.set("loop_manager", loop_manager)
    #                     return loop_config["pc_start"]
    #             else:
    #                 return pc_ + pc_increment_
    #         else:
    #             continue

    #     return pc_+pc_increment_

    # def decode(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     logger = logging.getLogger()
    #     i = args[0]
    #     j = args[1]
    #     ir = resource_pool_.get("ir_"+str(i)+"_"+str(j))
    #     ir1 = resource_pool_.get("ir1_"+str(i)+"_"+str(j))
    #     ir2 = resource_pool_.get("ir2_"+str(i)+"_"+str(j))
    #     ir3 = resource_pool_.get("ir3_"+str(i)+"_"+str(j))
    #     imm = resource_pool_.get("imm_"+str(i)+"_"+str(j))
    #     if not imm:
    #         if (ir.startswith("0001")):
    #             # REFI
    #             logger.debug("DECODE REFI DELAY [" + str(i) + ","+str(j)+"]")
    #             port = int(ir[4:6], 2)
    #             dimarch = int(ir[6:7], 2) and int(ir2[25:26], 2)
    #             if port >= 2:
    #                 res = str(i)+"_"+str(j) + "_shadow_REFI_out"+str(port-2)
    #                 if (dimarch):
    #                     res = res+"_dimarch"
    #                 resource_pool_.set(res, ir+ir1+ir2+ir3)
    #             else:
    #                 res = str(i)+"_"+str(j) + "_shadow_REFI_in"+str(port)
    #                 if (dimarch):
    #                     res = res+"_dimarch"
    #                 resource_pool_.set(res, ir+ir1+ir2)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             ext = int(ir[6:8], 2)
    #             pc = self.compute_npc(resource_pool_, i, j, pc, ext+1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("0100")):
    #             # DPU
    #             logger.debug("DECODE DPU DELAY [" + str(i) + ","+str(j)+"]")
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_DPU", ir)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("0101")):
    #             # SWB
    #             logger.debug("DECODE SWB DELAY [" + str(i) + ","+str(j)+"]")
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_SWB", ir)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1101")):
    #             # SRAM
    #             logger.debug("DECODE SRAM DELAY [" + str(i) + ","+str(j)+"]")
    #             resource_pool_.set(str(i)+"_"+str(j) +
    #                                "_shadow_SRAM", ir+ir1+ir2)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 3)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1010")):
    #             # RACCU
    #             logger.debug("DECODE RACCU DELAY [" + str(i) + ","+str(j)+"]")
    #             resource_pool_.set(str(i)+"_"+str(j) +
    #                                "_shadow_RACCU", ir+ir1)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, ext+1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1000")):
    #             # LOOP
    #             # logger.debug("DECODE LOOP DELAY [" + str(i) + ","+str(j)+"]")
    #             # resource_pool_.set(str(i)+"_"+str(j) + "_shadow_LOOP", ir+ir1)
    #             # pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             # loop_pc = pc
    #             # resource_pool_.set(str(i)+"_"+str(j) +
    #             #                    "_shadow_LOOP_pc", loop_pc)
    #             # pc = self.compute_npc(resource_pool_, i, j, pc, 1)
    #             # resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             # e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True))
    #             # event_pool_.post(e)

    #             logger.debug("DECODE LOOP [" + str(i) + ","+str(j)+"]")
    #             ext = int(ir[4], 2)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             loop_pc = pc
    #             pc = self.compute_npc(resource_pool_, i, j, pc, ext+1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_loop", [i, j, loop_pc, ir+ir1], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1111")):
    #             logger.debug("DECODE PERM [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, ext+1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_perm", [i, j, ir], 100, True)
    #             event_pool_.post(e)

    #     else:
    #         # issue shadow register
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_REFI_in0")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_REFI_in0", "")
    #             e = (clk_, "config_refi", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_REFI_in1")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_REFI_in1", "")
    #             e = (clk_, "config_refi", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_REFI_out0")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_REFI_out0", "")
    #             e = (clk_, "config_refi", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_REFI_out1")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_REFI_out1", "")
    #             e = (clk_, "config_refi", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(
    #             str(i)+"_"+str(j) + "_shadow_REFI_in0_dimarch")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) +
    #                                "_shadow_REFI_in0_dimarch", "")
    #             e = (clk_, "config_refi", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(
    #             str(i)+"_"+str(j) + "_shadow_REFI_out0_dimarch")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) +
    #                                "_shadow_REFI_out0_dimarch", "")
    #             e = (clk_, "config_refi", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_DPU")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_DPU", "")
    #             e = (clk_, "config_dpu", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_SWB")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_SWB", "")
    #             e = (clk_, "config_swb", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_SRAM")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_SRAM", "")
    #             e = (clk_, "config_sram", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_RACCU")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_RACCU", "")
    #             e = (clk_, "config_raccu", [i, j, si], 100, True)
    #             event_pool_.post(e)
    #         si = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_LOOP")
    #         loop_pc = resource_pool_.get(str(i)+"_"+str(j) + "_shadow_LOOP_pc")
    #         if (si):
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_LOOP", "")
    #             resource_pool_.set(str(i)+"_"+str(j) + "_shadow_LOOP_pc", -1)
    #             e = (clk_, "config_loop", [i, j, loop_pc, si], 100, True)
    #             event_pool_.post(e)

    #         print(ir, ir1, ir2, ir3)

    #         if (ir.startswith("0000")):
    #             # HALT
    #             pass
    #         elif (ir.startswith("0001")):
    #             # REFI
    #             logger.debug("DECODE REFI [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             ext = int(ir[6:8], 2)
    #             pc = self.compute_npc(resource_pool_, i, j, pc, ext+1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_refi", [i, j, ir+ir1+ir2+ir3], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("0100")):
    #             # DPU
    #             logger.debug("DECODE DPU [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_dpu", [i, j, ir], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("0101")):
    #             # SWB
    #             logger.debug("DECODE SWB [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_swb", [i, j, ir], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("0111")):
    #             # WAIT
    #             logger.debug("DECODE WAIT [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             cycles = int(ir[5:20], 2)
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+cycles+1, "fetch_"+str(i) +
    #                  "_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1100")):
    #             # ROUTE
    #             # For now, just ignore ROUTE instruction
    #             logger.debug("DECODE ROUTE [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 3)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1101")):
    #             # SRAM
    #             logger.debug("DECODE SRAM [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 3)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_sram", [i, j, ir+ir1+ir2], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1110")):
    #             # IO
    #             logger.debug("DECODE IO [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 3)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_io", [i, j, ir+ir1+ir2], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1010")):
    #             # RACCU
    #             logger.debug("DECODE RACCU [" + str(i) + ","+str(j)+"]")
    #             ext = int(ir[4], 2)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, ext+1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_raccu", [i, j, ir+ir1], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1000")):
    #             # LOOP
    #             logger.debug("DECODE LOOP [" + str(i) + ","+str(j)+"]")
    #             ext = int(ir[4], 2)
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             loop_pc = pc
    #             pc = self.compute_npc(resource_pool_, i, j, pc, ext+1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_loop", [i, j, loop_pc, ir+ir1], 100, True)
    #             event_pool_.post(e)
    #         elif (ir.startswith("1111")):
    #             # PERM
    #             logger.debug("DECODE PERM [" + str(i) + ","+str(j)+"]")
    #             pc = resource_pool_.get("pc_"+str(i)+"_"+str(j))
    #             pc = self.compute_npc(resource_pool_, i, j, pc, 1)
    #             resource_pool_.set("pc_"+str(i)+"_"+str(j), pc)
    #             e = (clk_+1, "fetch_"+str(i)+"_"+str(j), [i, j], 100, True)
    #             event_pool_.post(e)
    #             e = (clk_, "config_perm", [i, j, ir], 100, True)
    #             event_pool_.post(e)
    #         else:
    #             logger.fatal(
    #                 "Unknown Instruction in [" + str(i) + ","+str(j)+"]:" + ir)

    # def config_refi(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     instr = args[2]
    #     port = int(instr[4:6], 2)
    #     extra = int(instr[6:8], 2)
    #     init_addr_sd = int(instr[8:9], 2)
    #     init_addr = int(instr[9:15], 2)
    #     l1_iter = int(instr[15:21], 2)
    #     init_delay = int(instr[21:27], 2)

    #     l1_iter_sd = 0
    #     init_delay_sd = 0
    #     l1_step_sd = 0
    #     l1_step = 1
    #     l1_step_sign = 0
    #     l1_delay_sd = 0
    #     l1_delay = 0
    #     l2_iter_sd = 0
    #     l2_iter = 0
    #     l2_step = 0
    #     l2_delay_sd = 0
    #     l2_delay = 0
    #     l1_delay_ext = 0
    #     l2_iter_ext = 0
    #     l2_step_ext = 0
    #     dimarch = 0
    #     compress = 0
    #     mask = 0

    #     if (extra >= 1):
    #         l1_iter_sd = int(instr[27:28], 2)
    #         init_delay_sd = int(instr[28: 29], 2)
    #         l1_step_sd = int(instr[31], 2)
    #         l1_step = int(instr[32: 38], 2)
    #         l1_step_sign = int(instr[38], 2)
    #         l1_delay_sd = int(instr[39], 2)
    #         l1_delay = int(instr[40: 44], 2)
    #         l2_iter_sd = int(instr[44], 2)
    #         l2_iter = int(instr[45: 50], 2)
    #         l2_step = int(instr[50: 54], 2)
    #     if extra >= 2:
    #         l2_delay_sd = int(instr[58], 2)
    #         l2_delay = int(instr[59: 65], 2)
    #         l1_delay_ext = int(instr[71: 73], 2)
    #         l2_iter_ext = int(instr[73], 2)
    #         l2_step_ext = int(instr[74: 76], 2)
    #         dimarch = int(instr[79], 2)
    #         compress = int(instr[80], 2)
    #     if extra >= 3:
    #         mask = int(instr[81: 97], 2)
    #     is_read = 0
    #     if (port >= 2):
    #         is_read = 1
    #     reg_res = str(row)+"_"+str(col)+"_"
    #     if port >= 2:
    #         reg_res += "RF_out"+str(port-2)
    #     else:
    #         reg_res += "RF_in"+str(port)
    #     if (dimarch):
    #         reg_res += "_dimarch"
    #     raccu_reg = resource_pool_.get("raccu_reg")
    #     if init_addr_sd:
    #         init_addr = raccu_reg[row][col][init_addr]
    #     if l1_iter_sd:
    #         l1_iter = raccu_reg[row][col][l1_iter]
    #     if init_delay_sd:
    #         init_delay = raccu_reg[row][col][init_delay]
    #     if l1_step_sd:
    #         l1_step = raccu_reg[row][col][l1_step]
    #     if l1_delay_sd:
    #         l1_delay = raccu_reg[row][col][l1_delay]
    #     if l2_iter_sd:
    #         l2_iter = raccu_reg[row][col][l2_iter]
    #     if l2_delay_sd:
    #         l2_delay = raccu_reg[row][col][l2_delay]

    #     if l1_delay_ext:
    #         l1_delay = l1_delay_ext * 2**4 + l1_delay
    #     if l2_iter_ext:
    #         l2_iter = l2_iter_ext * 2**5 + l2_iter
    #     if l2_step_ext:
    #         l2_step = l2_step_ext * 2**4 + l2_step

    #     if l1_step_sign:
    #         l1_step = - l1_step

    #     t = clk_+init_delay
    #     a_l2 = init_addr
    #     for i in range(l2_iter+1):
    #         a_l1 = a_l2
    #         for j in range(l1_iter+1):
    #             if (is_read):
    #                 e = (t, "rf_read", [row, col, reg_res, a_l1], 100, True)
    #                 event_pool_.post(e)
    #             else:
    #                 e = (t, "rf_write", [
    #                      row, col, reg_res, a_l1, mask], 100, True)
    #                 event_pool_.post(e)
    #             a_l1 += l1_step
    #             if j < l1_iter:
    #                 t += l1_delay + 1
    #         a_l2 += l2_step
    #         if i < l2_iter:
    #             t += l2_delay + 1

    # def rf_read(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     res = args[2]
    #     addr = args[3]
    #     reg = resource_pool_.get("regfile")
    #     val = reg[row][col][addr]
    #     if "dimarch" in res:
    #         val = reg[row][col][addr*16:(addr+1)*16]
    #     resource_pool_.set(res, val)
    #     self.logger.notice("READ FROM RF(" + str(row) + "," +
    #                        str(col) + ")["+str(addr)+"] = " + str(val))

    # def rf_write(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     res = args[2]
    #     addr = args[3]
    #     mask = args[4]
    #     reg = resource_pool_.get("regfile")
    #     val = resource_pool_.get(res)
    #     val2 = reg[row][col][addr*16:(addr+1)*16]
    #     if "dimarch" in res:
    #         for i in range(16):
    #             if not (mask & (1 << i)):
    #                 val2[i] = val[i]
    #         reg[row][col][addr*16:(addr+1)*16] = val2
    #         resource_pool_.set("regfile", reg)
    #         self.logger.notice("WRITE TO RF(" + str(row) + "," +
    #                            str(col) + ")["+str(addr)+"] = " + str(val2))
    #     else:
    #         reg[row][col][addr] = val
    #         resource_pool_.set("regfile", reg)
    #         self.logger.notice("WRITE TO RF(" + str(row) + "," +
    #                            str(col) + ")["+str(addr)+"] = " + str(val))

    # def sram_read(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     # TODO: FIXME
    #     row = args[0]
    #     col = args[1]
    #     res = args[2]
    #     addr = args[3]
    #     input_buffer = resource_pool_.get("sram")
    #     val = input_buffer[addr]
    #     resource_pool_.set(res, val)
    #     self.logger.notice(
    #         "READ FROM INPUT_BUFFER["+str(addr)+"] = " + str(val))

    # def sram_write(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     res = args[2]
    #     addr = args[3]
    #     sram = resource_pool_.get("sram")
    #     val = resource_pool_.get(res)
    #     sram[row][col][addr] = val
    #     resource_pool_.set("sram", sram)
    #     self.logger.notice("WRITE TO SRAM(" + str(row) +
    #                        "," + str(col) + ")["+str(addr)+"] = " + str(val))

    # def io_read(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     res = args[2]
    #     addr = args[3]
    #     input_buffer = resource_pool_.get("input_buffer")
    #     val = input_buffer[addr]
    #     resource_pool_.set(res, val)

    #     iap = resource_pool_.get("IAP")
    #     iap.append([clk_, addr])
    #     resource_pool_.set("IAP", iap)

    #     self.logger.notice(
    #         "READ FROM INPUT_BUFFER["+str(addr)+"] = " + str(val))

    # def io_write(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     res = args[2]
    #     addr = args[3]
    #     mask = args[4]
    #     output_buffer = resource_pool_.get("output_buffer")
    #     val = resource_pool_.get(res)
    #     if addr in output_buffer:
    #         val2 = output_buffer[addr]
    #     else:
    #         val2 = [0]*16
    #     print(mask, res, val)
    #     for i in range(16):
    #         if not (mask & (1 << i)):
    #             val2[i] = val[i]

    #     output_buffer[addr] = val2
    #     resource_pool_.set("output_buffer", output_buffer)

    #     output_buffer_active = resource_pool_.get("output_buffer_active")
    #     output_buffer_active[addr] = True
    #     resource_pool_.set("output_buffer_active", output_buffer_active)

    #     oap = resource_pool_.get("OAP")
    #     oap.append([clk_, addr])
    #     resource_pool_.set("OAP", oap)

    #     self.logger.notice(
    #         "WRITE TO OUTPUT_BUFFER["+str(addr)+"] = " + str(val2))

    # def config_dpu(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     instr = args[2]
    #     mode = int(instr[4: 9], 2)
    #     saturation = int(instr[9: 10], 2)
    #     fixed = int(instr[10: 11], 2)
    #     acc_clear = int(instr[17: 25], 2)
    #     io_change = int(instr[25: 27], 2)
    #     negate_in0 = False
    #     negate_in1 = False
    #     absolute_out = False
    #     if (io_change == 1):
    #         negate_in0 = True
    #     elif (io_change == 2):
    #         negate_in1 = True
    #     elif (io_change == 3):
    #         absolute_out = True

    #     config = {"mode": mode, "saturation": saturation, "fixed": fixed, "acc_clear": acc_clear,
    #               "negate_in0": negate_in0, "negate_in1": negate_in1, "absolute_out": absolute_out, "immediate": True}

    #     e = (clk_+1, "post_dpu_config", [row, col, config], 100, True)
    #     event_pool_.post(e)

    # def post_dpu_config(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     config = args[2]

    #     resource_pool_.set("dpu_"+str(row)+"_"+str(col)+"_config", config)
    #     resource_pool_.set(str(row)+"_"+str(col)+"_DPU_acc0", 0)
    #     resource_pool_.set(str(row)+"_"+str(col)+"_DPU_acc1", 0)

    # def config_swb(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     i = args[0]
    #     j = args[1]
    #     instr = args[2]
    #     src_row = int(instr[5], 2)
    #     src_block = int(instr[6], 2)
    #     src_port = int(instr[7], 2)
    #     hb_index = int(instr[8:11], 2)
    #     v_index = int(instr[12:15], 2)
    #     src_row_str = str(src_row)
    #     src_col_str = str(j-2+hb_index)
    #     src_port_str = str(src_port)
    #     src_block_str = "RF_out"
    #     if (src_block == 1):
    #         src_block_str = "DPU_out"
    #     dest_row_str = str(i)
    #     dest_col_str = str(j)
    #     dest_port_str = "0"
    #     dest_block_str = "RF_in"
    #     if (v_index == 0):
    #         dest_port_str = "0"
    #         dest_block_str = "RF_in"
    #     elif v_index == 1:
    #         dest_port_str = "1"
    #         dest_block_str = "RF_in"
    #     elif v_index == 2:
    #         dest_port_str = "0"
    #         dest_block_str = "DPU_in"
    #     elif v_index == 3:
    #         dest_port_str = "1"
    #         dest_block_str = "DPU_in"
    #     elif v_index == 4:
    #         dest_port_str = "2"
    #         dest_block_str = "DPU_in"
    #     elif v_index == 5:
    #         dest_port_str = "3"
    #         dest_block_str = "DPU_in"
    #     connection_map = resource_pool_.get("connection_map")
    #     connection_map[dest_row_str + "_" + dest_col_str+"_"+dest_block_str +
    #                    dest_port_str] = src_row_str+"_"+src_col_str+"_"+src_block_str + src_port_str
    #     resource_pool_.set("connection_map", connection_map)

    # def config_sram(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     instr = args[2]
    #     is_write = int(instr[4:5], 2)
    #     init_addr = int(instr[5:12], 2)
    #     init_delay = int(instr[12:16], 2)
    #     l1_iter = int(instr[16:23], 2)
    #     l1_step = int(instr[23:31], 2)
    #     l1_delay = int(instr[31:37], 2)
    #     l2_iter = int(instr[37:44], 2)
    #     l2_step = int(instr[44: 52], 2)
    #     l2_delay = int(instr[52: 58], 2)
    #     init_addr_sd = int(instr[58:59], 2)

    #     if (l1_step >= 128):
    #         l1_step = l1_step - 256
    #     if (l2_step >= 128):
    #         l2_step = l2_step - 256

    #     reg_res = "0_"+str(col)+"_"
    #     if (is_write):
    #         reg_res += "SRAM_in"
    #     else:
    #         reg_res += "SRAM_out"

    #     raccu_reg = resource_pool_.get("raccu_reg")
    #     if init_addr_sd:
    #         init_addr = raccu_reg[row][col][init_addr]

    #     t = clk_+init_delay
    #     a_l2 = init_addr
    #     for i in range(l2_iter+1):
    #         a_l1 = a_l2
    #         for j in range(l1_iter+1):
    #             if (not is_write):
    #                 e = (t+2, "sram_read", [0, col, reg_res, a_l1], 100, True)
    #                 event_pool_.post(e)
    #             else:
    #                 e = (t+4, "sram_write", [0, col, reg_res, a_l1], 100, True)
    #                 event_pool_.post(e)
    #             a_l1 += l1_step
    #             if j < l1_iter:
    #                 t += l1_delay + 1
    #         a_l2 += l2_step
    #         t += l2_delay + 1

    # def config_io(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     instr = args[2]
    #     is_write = int(instr[4:5], 2)
    #     init_addr = int(instr[5:30], 2)
    #     init_delay = int(instr[30:36], 2)
    #     l1_iter = int(instr[36:43], 2)
    #     l1_step = int(instr[43:49], 2)
    #     l1_delay = int(instr[49:55], 2)
    #     mask = int(instr[55:71], 2)
    #     l2_iter = 0
    #     l2_step = 0
    #     l2_delay = 0
    #     init_addr_sd = int(instr[71:72], 2)
    #     init_delay_sd = int(instr[72:73], 2)
    #     init_iter_sd = int(instr[73:74], 2)
    #     init_step_sd = int(instr[74:75], 2)

    #     if (l1_step >= 128):
    #         l1_step = l1_step - 256
    #     if (l2_step >= 128):
    #         l2_step = l2_step - 256

    #     reg_res = ""
    #     if (is_write):
    #         reg_res += str(row)+"_"+str(col)+"_IO_in"
    #     else:
    #         reg_res += str(row)+"_"+str(col)+"_IO_out"

    #     raccu_reg = resource_pool_.get("raccu_reg")
    #     if init_addr_sd:
    #         init_addr = raccu_reg[row][col][init_addr]

    #     t = clk_+init_delay
    #     a_l2 = init_addr
    #     for i in range(l2_iter+1):
    #         a_l1 = a_l2
    #         for j in range(l1_iter+1):
    #             if (not is_write):
    #                 e = (t, "io_read", [0, col, reg_res, a_l1], 100, True)
    #                 event_pool_.post(e)
    #             else:
    #                 e = (t, "io_write", [
    #                      0, col, reg_res, a_l1, mask], 100, True)
    #                 event_pool_.post(e)
    #             a_l1 += l1_step
    #             if j < l1_iter:
    #                 t += l1_delay + 1
    #         a_l2 += l2_step
    #         t += l2_delay + 1

    # def config_raccu(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     instr = args[2]
    #     extra = int(instr[4], 2)
    #     mode = int(instr[5:9], 2)
    #     operand1_sd = int(instr[9], 2)
    #     operand1 = int(instr[10: 16], 2)
    #     operand2_sd = int(instr[16], 2)
    #     operand2 = int(instr[17:23], 2)
    #     if operand2 >= 2**6:
    #         operand2 = operand2 - 2**7
    #     result = int(instr[23:27], 2)
    #     if extra:
    #         operand1_ext = int(instr[27:46], 2)
    #     else:
    #         operand1_ext = 0
    #     operand1 = (operand1_ext << 6) + operand1
    #     if operand1 >= 2**24:
    #         operand1 = operand1 - 2**25

    #     raccu_reg = resource_pool_.get("raccu_reg")
    #     if operand1_sd:
    #         operand1 = raccu_reg[row][col][operand1]
    #     if operand2_sd:
    #         operand2 = raccu_reg[row][col][operand2]

    #     logging.debug("RACCU: mode=" + str(mode) + ",   " + str(operand1) +
    #                   " " + str(operand2) + " " + str(result))

    #     if mode == 1:
    #         raccu_reg[row][col][result] = operand1 + operand2
    #     elif mode == 2:
    #         raccu_reg[row][col][result] = operand1 - operand2
    #     elif mode == 3:
    #         raccu_reg[row][col][result] = operand1 // (2**operand2)
    #     elif mode == 4:
    #         raccu_reg[row][col][result] = operand1 * (2**operand2)
    #     elif mode == 5:
    #         raccu_reg[row][col][result] = operand1 * operand2
    #     else:
    #         self.logger.fatal("Unknown RACCU mode: " + str(mode) + "!")
    #         exit(1)

    #     resource_pool_.set("raccu_reg", raccu_reg)

    # def config_loop(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     pc_start = args[2]
    #     instr = args[3]
    #     raccu_reg = resource_pool_.get("raccu_reg")
    #     ext = int(instr[4], 2)
    #     loop_id = int(instr[5: 7], 2)
    #     pc_end = int(instr[7: 13], 2)
    #     start_sd = int(instr[13], 2)
    #     start = int(instr[14:20], 2)
    #     if start_sd:
    #         start = raccu_reg[row][col][start]
    #     iter_sd = int(instr[20], 2)
    #     iter = int(instr[21:27], 2)
    #     step_sd = 0
    #     step = 1
    #     if ext:
    #         step_sd = int(instr[27], 2)
    #         step = int(instr[28:34], 2)
    #         if step_sd:
    #             step = raccu_reg[row][col][step]

    #         iter_ext = int(instr[34:50], 2)
    #         iter = (iter_ext << 6) + iter
    #         if iter_sd:
    #             iter = raccu_reg[row][col][iter]

    #     if iter_sd:
    #         iter = raccu_reg[row][col][iter]

    #     if iter <= 0:
    #         pc = self.compute_npc(resource_pool_, row, col, pc_end, 1)
    #         resource_pool_.set("pc_" + str(row)+"_"+str(col), pc)
    #         return

    #     loop_manager = resource_pool_.get("loop_manager")
    #     loop_manager[row][col][loop_id]["enable"] = True
    #     loop_manager[row][col][loop_id]["pc_start"] = pc_start+ext+1
    #     loop_manager[row][col][loop_id]["pc_end"] = pc_end
    #     loop_manager[row][col][loop_id]["iter"] = iter
    #     loop_manager[row][col][loop_id]["step"] = step
    #     resource_pool_.set("loop_manager", loop_manager)

    #     raccu_reg[row][col][15-loop_id] = start
    #     resource_pool_.set("raccu_reg", raccu_reg)

    # def config_perm(self, clk_, event_pool_, resource_pool_, handler_pool_, args):
    #     row = args[0]
    #     col = args[1]
    #     instr = args[2]
    #     mode = int(instr[4:7], 2)
    #     block = int(instr[7: 9], 2)
    #     distance = int(instr[9: 25], 2)

    #     reg = resource_pool_.get("regfile")
    #     val = reg[row][col][block*16:(block+1)*16]
    #     new_val = [0]*16
    #     if mode == 0:
    #         # left shift padding 0
    #         for i in range(16):
    #             if i >= distance:
    #                 new_val[i] = val[i-distance]
    #     elif mode == 1:
    #         # right shift padding 0
    #         for i in range(16):
    #             if i < 16-distance:
    #                 new_val[i] = val[i+distance]
    #     elif mode == 2:
    #         # left shift padding with the last element
    #         for i in range(16):
    #             if i >= distance:
    #                 new_val[i] = val[i-distance]
    #             else:
    #                 new_val[i] = val[0]
    #     elif mode == 3:
    #         # right shift padding with the first element
    #         for i in range(16):
    #             if i < 16-distance:
    #                 new_val[i] = val[i+distance]
    #             else:
    #                 new_val[i] = val[15]
    #     elif mode == 4:
    #         # left rotate
    #         for i in range(16):
    #             new_val[i] = val[(i-distance) % 16]
    #     elif mode == 5:
    #         # right rotate
    #         for i in range(16):
    #             new_val[i] = val[(i+distance) % 16]
    #     else:
    #         self.logger.fatal("Unknown perm mode: " + str(mode) + "!")

    #     reg[row][col][block*16:(block+1)*16] = new_val
    #     resource_pool_.set("regfile", reg)
