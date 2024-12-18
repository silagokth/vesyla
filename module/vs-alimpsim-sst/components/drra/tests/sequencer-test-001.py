import sst

global_clock = "100MHz"
io_data_width = 256 # 256 bits
io_buffer_size = 1024 # 1024 bytes

# Initialize input and output buffers
input_buffer = sst.Component("input_buffer", "drra.IOBuffer")
output_buffer = sst.Component("output_buffer", "drra.IOBuffer")
buffer_params = {
    "clock": global_clock,
    "io_data_width": io_data_width,
    "access_time": "0ns",
}
input_buffer.addParams(buffer_params)
output_buffer.addParams(buffer_params)

# Initialize sequencer
seq_obj = sst.Component("sequencer", "drra.Sequencer")
seq_obj.addParams({
    "clock": global_clock,
    "assembly_program_path": "components/drra/tests/sequencer-test-001.bin",
    "cell_coordinates": [0, 0]
})

# Initialize slots
vec_adds = []
slot_controller_links = []
slot_input_buffer_links = []
slot_output_buffer_links = []
for i in range(16):
    # Initialize vector adder
    vec_adds.append(sst.Component("vec_add_" + str(i), "drra.VecAdd"))
    vec_adds[i].addParams({"clock": global_clock})

    # Connect vector adder to sequencer
    slot_controller_links.append(sst.Link("slot_controller_link_" + str(i)))
    slot_controller_links[i].connect((seq_obj, "slot_port" + str(i), "0ps"), (vec_adds[i], "controller_port", "0ps"))

    # Connect vector adder to input and output buffers
    slot_input_buffer_links.append(sst.Link("slot_input_buffer_link_" + str(i)))
    slot_input_buffer_links[i].connect((vec_adds[i], "input_buffer_port", "0ps"), (input_buffer, "slot_port" + str(i), "0ps"))
    slot_output_buffer_links.append(sst.Link("slot_output_buffer_link_" + str(i)))
    slot_output_buffer_links[i].connect((vec_adds[i], "output_buffer_port", "0ps"), (output_buffer, "slot_port" + str(i), "0ps"))
