## resource / swb

### swb [opcode=4]

Field | Position | Width | Default Value | Description
------|----------|-------|---------------|-------------------------
slot | [28, 25] | 4 |0| Slot number
option | [24, 23] | 2 |0| configuration option
channel | [22, 19] | 4 |0| Bus channel. Note: if the SWB is implemented by a crossbar, the channel is always equals to the target slot.
source | [18, 15] | 4 |0| Source slot.
target | [14, 11] | 4 |0| Target slot.

### route [opcode=5]

Field | Position | Width | Default Value | Description
------|----------|-------|---------------|-------------------------
slot | [28, 25] | 4 |0| Slot number
option | [24, 23] | 2 |0| configuration option
sr | [22, 22] | 1 |0| Send or receive. [0]:s; [1]:r;
source | [21, 18] | 4 |0| 1-hot encoded direction: E/N/W/S. If it's a receive instruction, the direction can only have 1 bit set to 1.
target | [17, 2] | 16 |0| 1-hot encoded slot number. If it's a send instruction, the slot can only have 1 bit set to 1.

### rep [opcode=0]

Field | Position | Width | Default Value | Description
------|----------|-------|---------------|-------------------------
slot | [28, 25] | 4 |0| Slot number
port | [24, 23] | 2 |0| The port number [0]:read narrow; [1]:read wide; [2]:write narrow; [3]:write wide;
level | [22, 19] | 4 |0| The level of the REP instruction. [0]: inner most level, [15]: outer most level.
iter | [18, 13] | 6 |0| level-1 iteration - 1.
step | [12, 7] | 6 |1| level-1 step
delay | [6, 1] | 6 |0| delay

### repx [opcode=1]

Field | Position | Width | Default Value | Description
------|----------|-------|---------------|-------------------------
slot | [28, 25] | 4 |0| Slot number
port | [24, 23] | 2 |0| The port number [0]:read narrow; [1]:read wide; [2]:write narrow; [3]:write wide;
level | [22, 19] | 4 |0| The level of the REP instruction. [0]: inner most level, [15]: outer most level.
iter | [18, 13] | 6 |0| level-1 iteration - 1.
step | [12, 7] | 6 |1| level-1 step
delay | [6, 1] | 6 |0| delay

### fsm [opcode=2]

Field | Position | Width | Default Value | Description
------|----------|-------|---------------|-------------------------
slot | [28, 25] | 4 |0| Slot number
port | [24, 23] | 2 |0| The port number
delay_0 | [22, 16] | 7 |0| Delay between state 0 and 1.
delay_1 | [15, 9] | 7 |0| Delay between state 1 and 2.
delay_2 | [8, 2] | 7 |0| Delay between state 2 and 3.