import sst

# Initialize sequencer
seq_obj = sst.Component("sequencer", "sequencer.Sequencer")
seq_obj.addParams({
    "assembly_program_path": "components/sequencer/tests/instr.bin",
    "cell_coordinates": [0, 0]
})

vec_adds = []
slot_links = []
for i in range(16):
    vec_adds.append(sst.Component("vec_add_" + str(i), "vec_add.VecAdd"))
    vec_adds[i].addParams({})
    slot_links.append(sst.Link("slot_link_" + str(i)))
    slot_links[i].connect((seq_obj, "slot_port" + str(i), "50ps"), (vec_adds[i], "controller_port", "50ps"))