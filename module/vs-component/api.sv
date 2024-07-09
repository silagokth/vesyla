package resource_swb_api;
    // Instruction: swb
    opcode_swb = 3'd4;
    typedef struct {
        logic[3:0] slot; // Slot number
        logic[1:0] option; // configuration option
        logic[3:0] channel; // Bus channel. Note: if the SWB is implemented by a crossbar, the channel is always equals to the target slot.
        logic[3:0] source; // Source slot.
        logic[3:0] target; // Target slot.
    }swb_t;
    function swb_t unpack_swb;
        input logic [27:0] instr;
        swb_t st;
        st.slot = instr[27:24];
        st.option = instr[23:22];
        st.channel = instr[21:18];
        st.source = instr[17:14];
        st.target = instr[13:10];
        return st;
    endfunction
    function logic [27:0] pack_swb;
        input swb_t st;
        logic [27:0] instr;
        instr[27:24] = st.slot;
        instr[23:22] = st.option;
        instr[21:18] = st.channel;
        instr[17:14] = st.source;
        instr[13:10] = st.target;
        return instr;
    endfunction
    
    // Instruction: route
    opcode_route = 3'd5;
    typedef struct {
        logic[3:0] slot; // Slot number
        logic[1:0] option; // configuration option
        logic sr; // Send or receive.
        logic[3:0] source; // 1-hot encoded direction: E/N/W/S. If it's a receive instruction, the direction can only have 1 bit set to 1.
        logic[15:0] target; // 1-hot encoded slot number. If it's a send instruction, the slot can only have 1 bit set to 1.
    }route_t;
    function route_t unpack_route;
        input logic [27:0] instr;
        route_t st;
        st.slot = instr[27:24];
        st.option = instr[23:22];
        st.sr = instr[21];
        st.source = instr[20:17];
        st.target = instr[16:1];
        return st;
    endfunction
    function logic [27:0] pack_route;
        input route_t st;
        logic [27:0] instr;
        instr[27:24] = st.slot;
        instr[23:22] = st.option;
        instr[21] = st.sr;
        instr[20:17] = st.source;
        instr[16:1] = st.target;
        return instr;
    endfunction
    
    // Instruction: rep
    opcode_rep = 3'd0;
    typedef struct {
        logic[3:0] slot; // Slot number
        logic[1:0] port; // The port number
        logic[3:0] level; // The level of the REP instruction. [0]: inner most level, [15]: outer most level.
        logic[5:0] iter; // level-1 iteration - 1.
        logic[5:0] step; // level-1 step
        logic[5:0] delay; // delay
    }rep_t;
    function rep_t unpack_rep;
        input logic [27:0] instr;
        rep_t st;
        st.slot = instr[27:24];
        st.port = instr[23:22];
        st.level = instr[21:18];
        st.iter = instr[17:12];
        st.step = instr[11:6];
        st.delay = instr[5:0];
        return st;
    endfunction
    function logic [27:0] pack_rep;
        input rep_t st;
        logic [27:0] instr;
        instr[27:24] = st.slot;
        instr[23:22] = st.port;
        instr[21:18] = st.level;
        instr[17:12] = st.iter;
        instr[11:6] = st.step;
        instr[5:0] = st.delay;
        return instr;
    endfunction
    
    // Instruction: repx
    opcode_repx = 3'd1;
    typedef struct {
        logic[3:0] slot; // Slot number
        logic[1:0] port; // The port number
        logic[3:0] level; // The level of the REP instruction. [0]: inner most level, [15]: outer most level.
        logic[5:0] iter; // level-1 iteration - 1.
        logic[5:0] step; // level-1 step
        logic[5:0] delay; // delay
    }repx_t;
    function repx_t unpack_repx;
        input logic [27:0] instr;
        repx_t st;
        st.slot = instr[27:24];
        st.port = instr[23:22];
        st.level = instr[21:18];
        st.iter = instr[17:12];
        st.step = instr[11:6];
        st.delay = instr[5:0];
        return st;
    endfunction
    function logic [27:0] pack_repx;
        input repx_t st;
        logic [27:0] instr;
        instr[27:24] = st.slot;
        instr[23:22] = st.port;
        instr[21:18] = st.level;
        instr[17:12] = st.iter;
        instr[11:6] = st.step;
        instr[5:0] = st.delay;
        return instr;
    endfunction
    
    // Instruction: fsm
    opcode_fsm = 3'd2;
    typedef struct {
        logic[3:0] slot; // Slot number
        logic[1:0] port; // The port number
        logic[6:0] delay_0; // Delay between state 0 and 1.
        logic[6:0] delay_1; // Delay between state 1 and 2.
        logic[6:0] delay_2; // Delay between state 2 and 3.
    }fsm_t;
    function fsm_t unpack_fsm;
        input logic [27:0] instr;
        fsm_t st;
        st.slot = instr[27:24];
        st.port = instr[23:22];
        st.delay_0 = instr[21:15];
        st.delay_1 = instr[14:8];
        st.delay_2 = instr[7:1];
        return st;
    endfunction
    function logic [27:0] pack_fsm;
        input fsm_t st;
        logic [27:0] instr;
        instr[27:24] = st.slot;
        instr[23:22] = st.port;
        instr[21:15] = st.delay_0;
        instr[14:8] = st.delay_1;
        instr[7:1] = st.delay_2;
        return instr;
    endfunction
    
endpackage