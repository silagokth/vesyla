#define {{name}} {{fingerprint}}
#define {{name}}_pkg {{fingerprint}}_pkg

{% if not already_defined %}
package {{fingerprint}}_pkg;
    {%- for p in parameters %}
    parameter {{p}} = {{parameters[p]}};
    {%- endfor %}

    typedef struct packed {
        logic mode;
        logic [15:0] n;
    } add_t;

    function static add_t unpack_add;
        input logic [23:0] instr;

        add_t add;

        add.mode = instr[23];
        add.n  = instr[22:7];

        return add;
    endfunction

    function static logic [23:0] pack_add;
        input add_t add;

        logic [23:0] instr;

        instr[23] = add.mode;
        instr[22:7]  = add.n;

        return instr;
    endfunction
endpackage

module {{fingerprint}}
import {{fingerprint}}_pkg::*;
(
    input  logic clk_0,
    input  logic rst_n_0,
    input  logic instr_en_0,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr_0,
    input  logic activate_0,
    output logic io_en_in,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_in,
    input  logic [IO_DATA_WIDTH-1:0] io_data_in,
    output logic io_en_out,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_out,
    output logic [IO_DATA_WIDTH-1:0] io_data_out
);

    logic clk, rst_n, instr_en, activate;
    logic [RESOURCE_INSTR_WIDTH-1:0] instr;

    assign clk = clk_0;
    assign rst_n = rst_n_0;
    assign instr_en = instr_en_0;
    assign instr = instr_0;
    assign activate = activate_0;

    // Function definition:
    typedef enum logic [1:0] { RESET, IDLE, COMPUTE_0, COMPUTE_1} state_t;
    state_t state, next_state;
    logic [RESOURCE_INSTR_WIDTH-1:0] instr_reg, instr_reg_next;
    logic [2:0] opcode;
    logic [23:0] payload;

    assign opcode = instr_reg[26:24];
    assign payload = instr_reg[23:0];

    // FSM:
    always_ff @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            state <= RESET;
            instr_reg <= 0;
        end else begin
            state <= next_state;
            instr_reg <= instr_reg_next;
        end
    end

    always_comb begin
        instr_reg_next = instr_reg;
        next_state = state;
        case (state)
            RESET: begin
                next_state <= IDLE;
            end
            IDLE: begin
                if (activate == 1) begin
                    add_t _add;
                    _add = unpack_add(payload);
                    if (_add.mode == 1) begin
                        next_state <= COMPUTE_0;
                    end
                end
                if (instr_en) begin
                    instr_reg_next <= instr;
                end
            end
            COMPUTE_0: begin
                next_state <= COMPUTE_1;
            end
            COMPUTE_1: begin
                next_state <= IDLE;
            end
        endcase
    end

    always_comb begin
        io_en_in = 0;
        io_addr_in = 0;
        io_en_out = 0;
        io_addr_out = 0;
        io_data_out = 0;
        case (state)
            COMPUTE_0: begin
                add_t _add;
                _add = unpack_add(payload);
                io_en_in = 1;
                io_addr_in = _add.n;
            end

            COMPUTE_1: begin
                add_t _add;
                _add = unpack_add(payload);
                for (int i = 0; i < 16; i = i + 1) begin : gen_block
                    io_data_out[16*i +: 16] = io_data_in[16*i +: 16] + 1;
                end
                io_en_out = 1;
                io_addr_out = _add.n;
            end
        endcase
    end
endmodule

{% endif %}