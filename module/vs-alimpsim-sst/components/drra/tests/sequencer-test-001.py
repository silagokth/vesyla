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
    "printFrequency": 1,
    "memory_file": "components/drra/tests/sequencer-test-001.mem",
    "backing": "mmap",
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

# Initialize VecAdd
vec_add = sst.Component("vec_add", "drra.VecAdd")
vec_add.addParams({
    "clock": global_clock,
    "slot_id": 1,
    "has_io_input_connection": 1,
    "has_io_output_connection": 1,
    "printFrequency": 1,
    "cell_coordinates": [0, 0]
})

# Connect VecAdd to Sequencer
slot_controller_link = sst.Link("slot_controller_link")
slot_controller_link.connect((seq_obj, "slot_port0", "0ps"), (vec_add, "controller_port", "0ps"))

# Connect VecAdd to IO Buffers
slot_input_buffer_link = sst.Link("slot_input_buffer_link")
slot_input_buffer_link.connect((input_buffer, "col_port0", "0ps"), (vec_add, "input_buffer_port", "0ps"))
slot_output_buffer_link = sst.Link("slot_output_buffer_link")
slot_output_buffer_link.connect((vec_add, "output_buffer_port", "0ps"), (output_buffer, "col_port0", "0ps"))