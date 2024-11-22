`define {{name}} {{fingerprint}}
`define {{name}}_pkg {{fingerprint}}_pkg

{% if not already_defined %}
package {{fingerprint}}_pkg;
    {% for p in parameters %}
    parameter {{p}} = {{parameters[p]}};
    {% endfor %}

    // Others:
    parameter OPCODE_HALT = 0;
    parameter OPCODE_WAIT = 1;
    parameter OPCODE_ACTIVATE = 2;

    typedef struct packed {
        logic [15:0] ports;
        logic [3:0]  mode;
        logic [7:0]  param;
    } activate_t;

    function static activate_t unpack_activate;
        input logic [27:0] instr;

        activate_t activate;

        activate.ports = instr[27:12];
        activate.mode  = instr[11:8];
        activate.param = instr[7:0];

        return activate;
    endfunction

    function static logic [27:0] pack_activate;
        input activate_t activate;

        logic [27:0] instr;

        instr[27:12] = activate.ports;
        instr[11:8]  = activate.mode;
        instr[7:0]   = activate.param;

        return instr;
    endfunction

    typedef struct {
        logic mode;
        logic [26:0] cycle;
    } wait_t;

    function static wait_t unpack_wait;
        input logic [27:0] instr;

        wait_t _wait;

        _wait.mode  = instr[27];
        _wait.cycle = instr[26:0];

        return _wait;
    endfunction

    function static logic [27:0] pack_wait;
        input wait_t _wait;

        logic [27:0] instr;

        instr[27]   = _wait.mode;
        instr[26:0] = _wait.cycle;

        return instr;
    endfunction

endpackage

module {{fingerprint}}
import {{fingerprint}}_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    input  logic call,
    output logic ret,
    output logic [NUM_SLOTS-1:0] instr_en,
    output logic [RESOURCE_INSTR_WIDTH-1:0] instr,
    output logic [NUM_SLOTS-1:0] activate,
    input  logic [INSTR_DATA_WIDTH-1:0] instr_data_in,
    input  logic [INSTR_ADDR_WIDTH-1:0] instr_addr_in,
    input  logic [INSTR_HOPS_WIDTH-1:0] instr_hops_in,
    input  logic instr_en_in,
    output logic [INSTR_DATA_WIDTH-1:0] instr_data_out,
    output logic [INSTR_ADDR_WIDTH-1:0] instr_addr_out,
    output logic [INSTR_HOPS_WIDTH-1:0] instr_hops_out,
    output logic instr_en_out
);

    // Parameter check:

    // Function definition:

    logic [27:0] instr_reg;
    logic [63:0][31:0] iram;
    logic [3:0] opcode;
    logic instr_type;
    logic [3:0] slot;
    logic [23:0] payload;
    logic [5:0] pc, pc_next;
    logic [16:0] wait_counter, wait_counter_next;

    typedef enum logic [1:0] { RESET, IDLE, DECODE, WAIT} state_t;
    state_t state, next_state;

    assign instr_type = iram[pc][31];
    assign opcode = iram[pc][30:28];
    assign instr_reg = iram[pc][27:0];
    assign payload = iram[pc][23:0];
    assign slot = iram[pc][27:24];

    always_ff @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            instr_data_out <= 0;
            instr_addr_out <= 0;
            instr_hops_out <= 0;
            instr_en_out <= 0;
            for (int i=0; i<64; i++) begin
                iram[i] = 0;
            end
        end else begin
            if (instr_en_in) begin
                if (instr_hops_in == 0) begin
                    iram[instr_addr_in] = instr_data_in;
                    instr_data_out = 0;
                    instr_addr_out = 0;
                    instr_hops_out = 0;
                    instr_en_out = 0;
                end else begin
                    instr_data_out = instr_data_in;
                    instr_addr_out = instr_addr_in;
                    instr_hops_out = instr_hops_in - 1;
                    instr_en_out = instr_en_in;
                end
            end else begin
                instr_data_out = 0;
                instr_addr_out = 0;
                instr_hops_out = 0;
                instr_en_out = 0;
            end
        end
    end

    // FSM:
    always_ff @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            state <= RESET;
            pc <= 0;
            wait_counter <= 0;
        end else begin
            state <= next_state;
            pc <= pc_next;
            wait_counter <= wait_counter_next;
        end
    end

    always_comb begin
        next_state = state;
        pc_next = pc+1;
        wait_counter_next = wait_counter;
        case (state)
            RESET: begin
                next_state = IDLE;
                pc_next = 0;
            end
            IDLE: begin
                if (call == 1) begin
                    next_state = DECODE;
                    pc_next = 0;
                end else begin
                    next_state = IDLE;
                    pc_next = 0;
                end
            end
            DECODE: begin
                if (instr_type == 0 && opcode == OPCODE_HALT) begin
                    pc_next = 0;
                    next_state = IDLE;
                end else if (instr_type == 0 && opcode == OPCODE_WAIT) begin
                    wait_t _wait;
                    _wait = unpack_wait(instr_reg);
                    if (_wait.cycle != 0) begin
                        pc_next = pc;
                        next_state = WAIT;
                        wait_counter_next = _wait.cycle;
                    end
                end
            end
            WAIT: begin
                wait_counter_next = wait_counter - 1;
                if (wait_counter == 1) begin
                    next_state = DECODE;
                end else begin
                    next_state = WAIT;
                    pc_next = pc;
                end
            end
        endcase
    end

    always_comb begin
        ret = 0;
        for (int i = 0; i < NUM_SLOTS; i = i + 1) begin
            instr_en[i] = 0;
            activate[i] = 0;
        end
        instr = 0;
        case (state)
            DECODE: begin
                if (instr_type == 0) begin
                    if (opcode == OPCODE_ACTIVATE) begin
                        activate_t _activate;
                        _activate = unpack_activate(instr_reg);
                        activate = _activate.ports;
                    end else if (opcode == OPCODE_HALT) begin
                        ret = 1;
                    end
                end else begin
                    instr = {opcode, payload};
                    instr_en[slot] = 1;
                end
            end
        endcase
    end
endmodule

{% endif %}
