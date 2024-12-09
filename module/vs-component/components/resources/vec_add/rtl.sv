`define {{name}} {{fingerprint}}
`define {{name}}_pkg {{fingerprint}}_pkg

{% if not already_defined %}
package {{fingerprint}}_pkg;
    {% for p in parameters %}
    parameter {{p}} = {{parameters[p]}};
    {% endfor %}

    {% set payload_bitwidth = isa.format.instr_bitwidth - isa.format.instr_type_bitwidth - isa.format.instr_opcode_bitwidth - isa.format.instr_slot_bitwidth %}
    {% for instr in isa.instructions %}
    typedef struct packed {
        {% for segment in instr.segments %}
        {% if segment.bitwidth == 1 %}
        logic _{{segment.name}};
        {% else %}
        logic [{{segment.bitwidth-1}}:0] _{{segment.name}};
        {% endif %}
        {% endfor %}
    } {{instr.name}}_t;

    function static {{instr.name}}_t unpack_{{instr.name}};
        input logic [{{payload_bitwidth - 1}}:0] instr;
        {{instr.name}}_t _{{instr.name}};
        {% set index=payload_bitwidth -1 %}
        {% for segment in instr.segments %}
        {% if segment.bitwidth==1 %}
        _{{instr.name}}._{{segment.name}} = instr[{{index}}];
        {% else %}
        _{{instr.name}}._{{segment.name}}  = instr[{{index}}:{{index-segment.bitwidth+1}}];
        {% endif %}
        {% set index=index-segment.bitwidth %}
        {% endfor %}
        return _{{instr.name}};
    endfunction

    function static logic [{{ payload_bitwidth - 1 }}:0] pack_{{instr.name}};
        input {{instr.name}}_t _{{instr.name}};
        logic [{{ payload_bitwidth - 1 }}:0] instr;

        {% set index=payload_bitwidth -1 %}
        {% for segment in instr.segments %}
        {% if segment.bitwidth==1 %}
        instr[{{index}}] = _{{instr.name}}._{{segment.name}};
        {% else %}
        instr[{{index}}:{{index-segment.bitwidth+1}}] = _{{instr.name}}._{{segment.name}};
        {% endif %}
        {% set index=index-segment.bitwidth %}
        {% endfor %}
        return instr;
    endfunction
    {% endfor %}
endpackage

module {{fingerprint}}
import {{fingerprint}}_pkg::*;
(
    input  logic clk_0,
    input  logic rst_n_0,
    input  logic instr_en_0,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr_0,
    input  logic activate_0[FSM_PER_SLOT-1:0],
    input  logic [WORD_BITWIDTH-1:0] word_data_in_0,
    output logic [WORD_BITWIDTH-1:0] word_data_out_0,
    input  logic [BULK_BITWIDTH-1:0] bulk_data_in_0,
    output logic [BULK_BITWIDTH-1:0] bulk_data_out_0,
    output logic io_en_in,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_in,
    input  logic [IO_DATA_WIDTH-1:0] io_data_in,
    output logic io_en_out,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_out,
    output logic [IO_DATA_WIDTH-1:0] io_data_out
);

    assign word_data_out_0 = 0;
    assign bulk_data_out_0 = 0;

    logic clk, rst_n, instr_en, activate;
    logic [RESOURCE_INSTR_WIDTH-1:0] instr;

    assign clk = clk_0;
    assign rst_n = rst_n_0;
    assign instr_en = instr_en_0;
    assign instr = instr_0;
    assign activate = activate_0[0];

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
                    if (_add._en == 1) begin
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
                io_addr_in = _add._addr;
            end

            COMPUTE_1: begin
                add_t _add;
                _add = unpack_add(payload);
                for (int i = 0; i < 16; i = i + 1) begin : gen_block
                    io_data_out[16*i +: 16] = io_data_in[16*i +: 16] + 1;
                end
                io_en_out = 1;
                io_addr_out = _add._addr;
            end
        endcase
    end
endmodule

{% endif %}
