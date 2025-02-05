import sst
import os

script_path = os.path.abspath(__file__)
script_path = os.path.dirname(script_path)

global_clock = "100MHz"

# Initialize sequencer
print("[SST SIM] - Initializing sequencer")
seq_obj = sst.Component("sequencer", "drra.Sequencer")
seq_obj.addParams(
    {
        "clock": global_clock,
        "assembly_program_path": script_path
        + "/refFiles/switchboxFSMSwitching_asm.bin",
        "printFrequency": 1,
        "cell_coordinates": [0, 0],
    }
)

# Initialize SwitchBox
print("[SST SIM] - Initializing SwitchBox")
switch_box = sst.Component("switch_box", "drra.Switchbox")
switch_box.addParams(
    {
        "clock": global_clock,
        "cell_coordinates": [0, 0],
        "printFrequency": 1,
        "slot_id": 0,
    }
)

# Connect slots to controller
print("[SST SIM] - Connecting components")
for index, slot in enumerate([switch_box]):
    slot_controller_link = sst.Link(f"slot_{index}_controller_link")
    slot_controller_link.connect(
        (seq_obj, f"slot_port{index}", "0ps"), (slot, "controller_port", "0ps")
    )

print("[SST SIM] - Simulation is starting")
