import sst
import os

script_path = os.path.abspath(__file__)
script_path = os.path.dirname(script_path)

# Global parameters
global_clock = str(100e6 * 10) + "Hz"  # 10x the real clock frequency (for priorities)
io_data_width = 256
io_buffer_size = 1024
word_bitwidth = 16
instr_bitwidth = 32
instr_type_bitwidth = 1
instr_opcode_width = 3
instr_slot_width = 4

# Initialize input and output buffers
print("[SST SIM] - Initializing input and output buffers")
input_buffer = sst.Component("input_buffer", "drra.IOBuffer")
output_buffer = sst.Component("output_buffer", "drra.IOBuffer")
buffer_params = {
    "clock": global_clock,
    "io_data_width": io_data_width,
    "access_time": "0ns",
    "printFrequency": 1,
    "backing": "mfile",
}
input_buffer.addParams(buffer_params)
input_buffer_params = {
    "memory_file": script_path + "/refFiles/example0_drra_template_inputBuffer.bin",
}
input_buffer.addParams(input_buffer_params)
output_buffer.addParams(buffer_params)
output_buffer_params = {
    "memory_file": script_path + "/refFiles/example0_drra_template_outputBuffer.bin",
}
output_buffer.addParams(output_buffer_params)

# CELL 0 0

# Controller: Sequencer
print("[SST SIM] - Initializing sequencer")
seq_obj_0_0 = sst.Component("seq_0_0", "drra.Sequencer")
seq_obj_0_0_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "assembly_program_path": script_path + "/refFiles/example0_drra_template_asm.bin",
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 0],
    "num_slots": 16,
}
seq_obj_0_0.addParams(seq_obj_0_0_params)

# Slot 0: SWB
print("[SST SIM] - Initializing SWB")
swb_0_0 = sst.Component("swb_0_0", "drra.Switchbox")
swb_0_0_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 0],
    "slot_id": 0,
    "number_of_fsms": 4,
    "num_slots": 16,
}
swb_0_0.addParams(swb_0_0_params)

# Slot 1: IOSRAM
print("[SST SIM] - Initializing IOSRAM")
iosram_0_0 = sst.Component("iosram_0_0", "drra.IOSRAM")
iosram_0_0_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 0],
    "slot_id": 1,
    "resource_size": 2,
    "has_io_input_connection": 1,
    "access_time": "0ns",
    "backing": "malloc",
    "backing_size_unit": "1MiB",
}
iosram_0_0.addParams(iosram_0_0_params)

cell_0_0_slots = [swb_0_0, iosram_0_0]
cell_0_0_slots_params = [swb_0_0_params, iosram_0_0_params]

# CELL 0 1

# Controller: Sequencer
print("[SST SIM] - Initializing sequencer")
seq_obj_0_1 = sst.Component("seq_0_1", "drra.Sequencer")
seq_obj_0_1_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "assembly_program_path": script_path + "/refFiles/example0_drra_template_asm.bin",
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 1],
    "num_slots": 16,
}
seq_obj_0_1.addParams(seq_obj_0_1_params)

# Slot 0: SWB
print("[SST SIM] - Initializing SWB")
swb_0_1 = sst.Component("swb_0_1", "drra.Switchbox")
swb_0_1_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 1],
    "slot_id": 0,
    "number_of_fsms": 4,
    "num_slots": 16,
}
swb_0_1.addParams(swb_0_1_params)

# Slot 1 to 3: Register Files
print("[SST SIM] - Initializing Register Files")
rfs_0_1 = []
rfs_0_1_params = []
for i in range(3):
    rfs_0_1.append(sst.Component("rf_0_1_" + str(i), "drra.RegisterFile"))
    rfs_0_1_params.append(
        {
            "clock": global_clock,
            "printFrequency": 1,
            "instr_bitwidth": instr_bitwidth,
            "instr_type_bitwidth": instr_type_bitwidth,
            "instr_opcode_width": instr_opcode_width,
            "instr_slot_width": instr_slot_width,
            "io_data_width": io_data_width,
            "word_bitwidth": word_bitwidth,
            "cell_coordinates": [0, 1],
            "slot_id": i + 1,
            "register_file_size": 256,
        }
    )
    rfs_0_1[i].addParams(rfs_0_1_params[i])

# Slot 4: DPU
print("[SST SIM] - Initializing DPU")
dpu_0_1 = sst.Component("dpu_0_1", "drra.DPU")
dpu_0_1_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 1],
    "slot_id": 4,
    "resource_size": 2,
}
dpu_0_1.addParams(dpu_0_1_params)

cell_0_1_slots = [swb_0_1, *rfs_0_1, dpu_0_1]
cell_0_1_slots_params = [swb_0_1_params, *rfs_0_1_params, dpu_0_1_params]

# CELL 0 2

# Controller: Sequencer
print("[SST SIM] - Initializing sequencer")
seq_obj_0_2 = sst.Component("seq_0_2", "drra.Sequencer")
seq_obj_0_2_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "assembly_program_path": script_path + "/refFiles/example0_drra_template_asm.bin",
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 2],
    "num_slots": 16,
}
seq_obj_0_2.addParams(seq_obj_0_2_params)

# Slot 0: SWB
print("[SST SIM] - Initializing SWB")
swb_0_2 = sst.Component("swb_0_2", "drra.Switchbox")
swb_0_2_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 2],
    "slot_id": 0,
    "number_of_fsms": 4,
    "num_slots": 16,
}
swb_0_2.addParams(swb_0_2_params)

# Slot 1: IOSRAM
print("[SST SIM] - Initializing IOSRAM")
iosram_0_2 = sst.Component("iosram_0_2", "drra.IOSRAM")
iosram_0_2_params = {
    "clock": global_clock,
    "printFrequency": 1,
    "instr_bitwidth": instr_bitwidth,
    "instr_type_bitwidth": instr_type_bitwidth,
    "instr_opcode_width": instr_opcode_width,
    "instr_slot_width": instr_slot_width,
    "io_data_width": io_data_width,
    "word_bitwidth": word_bitwidth,
    "cell_coordinates": [0, 2],
    "slot_id": 1,
    "resource_size": 2,
    "has_io_input_connection": 1,
    "access_time": "0ns",
    "backing": "malloc",
    "backing_size_unit": "1MiB",
}
iosram_0_2.addParams(iosram_0_2_params)

cell_0_2_slots = [swb_0_2, iosram_0_2]
cell_0_2_slots_params = [swb_0_2_params, iosram_0_2_params]

# CONNECTIONS


def get_resource_size(slot_params):
    if "resource_size" in slot_params:
        return slot_params["resource_size"]
    else:
        return 1


for cell_index, cell_slots in enumerate(
    [cell_0_0_slots, cell_0_1_slots, cell_0_2_slots]
):
    cell_slots_params = [
        cell_0_0_slots_params,
        cell_0_1_slots_params,
        cell_0_2_slots_params,
    ]
    seqs = [seq_obj_0_0, seq_obj_0_1, seq_obj_0_2]
    swbs = [swb_0_0, swb_0_1, swb_0_2]
    # Slot connections
    for index, slot in enumerate(cell_slots):
        # Connect sequencer to all slots
        for i in range(get_resource_size(cell_slots_params[cell_index][index])):
            seq_link = sst.Link(f"link_seq_0_{cell_index}_slot{index + i}")
            seq_link.connect(
                (seqs[cell_index], f"slot_port{index + i}", "0ns"),
                (slot, f"controller_port{i}", "0ns"),
            )
            print(
                f"[SST SIM] - Connected sequencer ({cell_index}) to slot {index+i} ({slot.getType()} ({index}))"
            )

            # Connect slots to swb
            if index != 0:  # all except swb
                swb_link = sst.Link(f"link_swb_0_{cell_index}_slot{index + i}")
                swb_link.connect(
                    (swbs[cell_index], f"slot_port{index + i}", "0ns"),
                    (slot, f"data_port{i}", "0ns"),
                )
                print(
                    f"[SST SIM] - Connected switchbox ({cell_index}) to slot {index+i} ({slot.getType()} ({index}))"
                )

# Connect cells SWB to each other
swb_0_0_to_0_1 = sst.Link("link_swb_0_0_to_0_1")
swb_0_1_to_0_2 = sst.Link("link_swb_0_1_to_0_2")
swb_0_0_to_0_1.connect((swb_0_0, "cell_port7", "0ns"), (swb_0_1, "cell_port1", "0ns"))
swb_0_1_to_0_2.connect((swb_0_1, "cell_port7", "0ns"), (swb_0_2, "cell_port1", "0ns"))

# Connect IOSRAMs to input and output buffers
iosram_0_0_to_input_buffer = sst.Link("link_iosram_0_0_to_input_buffer")
iosram_0_0_to_input_buffer.connect(
    (iosram_0_0, "io_port", "0ns"), (input_buffer, "col_port0", "0ns")
)
iosram_0_2_to_output_buffer = sst.Link("link_iosram_0_2_to_output_buffer")
iosram_0_2_to_output_buffer.connect(
    (iosram_0_2, "io_port", "0ns"), (output_buffer, "col_port0", "0ns")
)
