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

    parameter WORD_ADDR_WIDTH = 6;
    parameter BULK_ADDR_WIDTH = 2;
    parameter RF_DEPTH = 64;

endpackage

module {{fingerprint}}
import {{fingerprint}}_pkg::*;
(
    input  logic clk_0,
    input  logic rst_n_0,
    input  logic instr_en_0,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr_0,
    input  logic [3:0] activate_0,
    input  logic [WORD_BITWIDTH-1:0] word_data_in_0,
    output logic [WORD_BITWIDTH-1:0] word_data_out_0,
    input  logic [BULK_BITWIDTH-1:0] bulk_data_in_0,
    output logic [BULK_BITWIDTH-1:0] bulk_data_out_0
);

    logic clk, rst_n;
    assign clk = clk_0;
    assign rst_n = rst_n_0;

    logic [RF_DEPTH-1:0][WORD_BITWIDTH-1:0] memory, memory_next;
    logic bulk_w_en, bulk_r_en, word_w_en, word_r_en;
    logic [BULK_ADDR_WIDTH-1:0] bulk_w_addr, bulk_r_addr;
    logic [WORD_ADDR_WIDTH-1:0] word_w_addr, word_r_addr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            memory <= '{default: 0};
        end
        else begin
            memory <= memory_next;
        end
    end

    
    parameter WORD_PER_BULK = 2**(WORD_ADDR_WIDTH-BULK_ADDR_WIDTH);
    logic [WORD_ADDR_WIDTH-BULK_ADDR_WIDTH-1:0] offset_w_addr, offset_r_addr;
    always_comb begin
        memory_next = memory;
        bulk_data_out_0 = 0;
        word_data_out_0 = 0;
        if (bulk_w_en) begin
            for (int i=0; i<WORD_PER_BULK; i++) begin
                offset_w_addr = i;
                memory_next[{bulk_w_addr, offset_w_addr}] = bulk_data_in_0[i*WORD_PER_BULK +:WORD_PER_BULK];
            end
        end
        if (word_w_en) begin
            if ((!bulk_w_en) || (bulk_w_addr != word_w_addr[WORD_ADDR_WIDTH-1:WORD_ADDR_WIDTH-1-BULK_ADDR_WIDTH])) begin
                memory_next[word_w_addr] = word_data_in_0;
            end
        end
        if (bulk_r_en) begin
            for (int i=0; i<WORD_PER_BULK; i++) begin
                offset_r_addr = i;
                bulk_data_out_0[i*WORD_PER_BULK +:WORD_PER_BULK] = memory[{bulk_r_addr, offset_r_addr}];
            end
        end
        if (word_r_en) begin
            word_data_out_0 = memory[word_r_addr];
        end

    end

    logic [INSTR_OPCODE_BITWIDTH-1:0] opcode;
    logic [RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0] payload;

    assign opcode = instr_0[RESOURCE_INSTR_WIDTH-1:RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH];
    assign payload = instr_0[RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0];
    
    dsu_t dsu;
    rep_t rep;
    repx_t repx;

    logic [WORD_ADDR_WIDTH-1:0] addr_0_0, addr_0_1;
    logic [BULK_ADDR_WIDTH-1:0] addr_0_2, addr_0_3;
    logic dsu_valid;
    logic rep_valid;
    logic repx_valid;
    logic port_enable_0_0, port_enable_0_1, port_enable_0_2, port_enable_0_3;
    logic [WORD_ADDR_WIDTH-1:0] port_addr_0_0, port_addr_0_1;
    logic [BULK_ADDR_WIDTH-1:0] port_addr_0_2, port_addr_0_3;

    assign dsu_valid = instr_en_0 && (opcode == 6);
    assign rep_valid = instr_en_0 && (opcode == 0);
    assign repx_valid = instr_en_0 && (opcode == 1);
    assign dsu = dsu_valid ? unpack_dsu(payload) : '{default: 0};
    assign rep = rep_valid ? unpack_rep(payload) : '{default: 0};
    assign repx = repx_valid ? unpack_repx(payload) : '{default: 0};

    logic [WORD_ADDR_WIDTH-1:0] step_0_0, step_0_1;
    logic [WORD_ADDR_WIDTH-1:0] delay_0_0, delay_0_1;
    logic [WORD_ADDR_WIDTH-1:0] iter_0_0, iter_0_1;
    logic [WORD_ADDR_WIDTH-1:0] init_addr_0_0, init_addr_0_1;

    logic [BULK_ADDR_WIDTH-1:0] step_0_2, step_0_3;
    logic [BULK_ADDR_WIDTH-1:0] delay_0_2, delay_0_3;
    logic [BULK_ADDR_WIDTH-1:0] iter_0_2, iter_0_3;
    logic [BULK_ADDR_WIDTH-1:0] init_addr_0_2, init_addr_0_3;

    assign step_0_0 = rep._step[WORD_ADDR_WIDTH-1:0];
    assign delay_0_0 = rep._delay[WORD_ADDR_WIDTH-1:0];
    assign iter_0_0 = rep._iter[WORD_ADDR_WIDTH-1:0];
    assign init_addr_0_0 = dsu._init_addr[WORD_ADDR_WIDTH-1:0];
    assign step_0_1 = rep._step[WORD_ADDR_WIDTH-1:0];
    assign delay_0_1 = rep._delay[WORD_ADDR_WIDTH-1:0];
    assign iter_0_1 = rep._iter[WORD_ADDR_WIDTH-1:0];
    assign init_addr_0_1 = dsu._init_addr[WORD_ADDR_WIDTH-1:0];
    assign step_0_2 = rep._step[BULK_ADDR_WIDTH-1:0];
    assign delay_0_2 = rep._delay[BULK_ADDR_WIDTH-1:0];
    assign iter_0_2 = rep._iter[BULK_ADDR_WIDTH-1:0];
    assign init_addr_0_2 = dsu._init_addr[BULK_ADDR_WIDTH-1:0];
    assign step_0_3 = rep._step[BULK_ADDR_WIDTH-1:0];
    assign delay_0_3 = rep._delay[BULK_ADDR_WIDTH-1:0];
    assign iter_0_3 = rep._iter[BULK_ADDR_WIDTH-1:0];
    assign init_addr_0_3 = dsu._init_addr[BULK_ADDR_WIDTH-1:0];
    
    agu #(
      .ADDRESS_WIDTH(WORD_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
    ) aug_0_0 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[0]),
      .load_initial(dsu_valid & dsu._port == 0),
      .load_level(rep_valid & rep._port == 0),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep._level[1:0]),
      .step(step_0_0),
      .delay(delay_0_0),
      .iterations(iter_0_0),
      .initial_address(init_addr_0_0),
      .address_valid(word_w_en),
      .address(word_w_addr)
    );

    agu #(
      .ADDRESS_WIDTH(WORD_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
    ) aug_0_1 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[1]),
      .load_initial(dsu_valid & dsu._port == 1),
      .load_level(rep_valid & rep._port == 1),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep._level[1:0]),
      .step(step_0_1),
      .delay(delay_0_1),
      .iterations(iter_0_1),
      .initial_address(init_addr_0_1),
      .address_valid(word_r_en),
      .address(word_r_addr)
  );

  agu #(
      .ADDRESS_WIDTH(BULK_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
  ) agu_0_2 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[2]),
      .load_initial(dsu_valid & dsu._port == 2),
      .load_level(rep_valid & rep._port == 2),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep._level[1:0]),
      .step(step_0_2),
      .delay(delay_0_2),
      .iterations(iter_0_2),
      .initial_address(init_addr_0_2),
      .address_valid(bulk_w_en),
      .address(bulk_w_addr)
  );

  agu #(
      .ADDRESS_WIDTH(BULK_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
  ) agu_0_3 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[3]),
      .load_initial(dsu_valid & dsu._port == 3),
      .load_level(rep_valid & rep._port == 3),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep._level[1:0]),
      .step(step_0_3),
      .delay(delay_0_3),
      .iterations(iter_0_3),
      .initial_address(init_addr_0_3),
      .address_valid(bulk_r_en),
      .address(bulk_r_addr)
  );

endmodule

{% endif %}