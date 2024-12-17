import sst

global_clock = "100MHz"
io_data_width = 256 # 256 bits
io_buffer_size = 1024 # 1024 bytes

# Initialize sequencer
seq_obj = sst.Component("sequencer", "sequencer.Sequencer")
seq_obj.addParams({
    "clock": global_clock,
    "assembly_program_path": "components/sequencer/tests/sequencer-test-001.bin",
    "cell_coordinates": [0, 0]
})

# Initialize slots
vec_adds = []
slot_controller_links = []
for i in range(16):
    vec_adds.append(sst.Component("vec_add_" + str(i), "vec_add.VecAdd"))
    vec_adds[i].addParams({"clock": global_clock})
    slot_controller_links.append(sst.Link("slot_controller_link_" + str(i)))
    slot_controller_links[i].connect((seq_obj, "slot_port" + str(i), "0ps"), (vec_adds[i], "controller_port", "0ps"))
