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

    parameter SRAM_ADDR_WIDTH = 6;
    parameter DEPTH = 64;
    parameter WIDTH = 256;

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
    output logic [BULK_BITWIDTH-1:0] bulk_data_out_0,
    input  logic clk_1,
    input  logic rst_n_1,
    input  logic instr_en_1,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr_1,
    input  logic [3:0] activate_1,
    input  logic [WORD_BITWIDTH-1:0] word_data_in_1,
    output logic [WORD_BITWIDTH-1:0] word_data_out_1,
    input  logic [BULK_BITWIDTH-1:0] bulk_data_in_1,
    output logic [BULK_BITWIDTH-1:0] bulk_data_out_1,
    input  logic clk_2,
    input  logic rst_n_2,
    input  logic instr_en_2,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr_2,
    input  logic [3:0] activate_2,
    input  logic [WORD_BITWIDTH-1:0] word_data_in_2,
    output logic [WORD_BITWIDTH-1:0] word_data_out_2,
    input  logic [BULK_BITWIDTH-1:0] bulk_data_in_2,
    output logic [BULK_BITWIDTH-1:0] bulk_data_out_2,
    input  logic clk_3,
    input  logic rst_n_3,
    input  logic instr_en_3,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr_3,
    input  logic [3:0] activate_3,
    input  logic [WORD_BITWIDTH-1:0] word_data_in_3,
    output logic [WORD_BITWIDTH-1:0] word_data_out_3,
    input  logic [BULK_BITWIDTH-1:0] bulk_data_in_3,
    output logic [BULK_BITWIDTH-1:0] bulk_data_out_3,
    output logic io_en_out,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_out,
    output logic [IO_DATA_WIDTH-1:0] io_data_out
);
    // Useless outputs:
    assign word_data_out_0 = 0;
    assign word_data_out_1 = 0;
    assign word_data_out_2 = 0;
    assign word_data_out_3 = 0;
    assign bulk_data_out_0 = 0;
    assign bulk_data_out_1 = 0;
    assign bulk_data_out_2 = 0;
    assign bulk_data_out_3 = 0;

    logic clk, rst_n;
    assign clk = clk_0;
    assign rst_n = rst_n_0;

    logic [INSTR_OPCODE_BITWIDTH-1:0] opcode_0_1, opcode_0_3, opcode_1_2;
    logic [RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0] payload_0_1, payload_0_3, payload_1_2;

    assign opcode_0_1 = instr_0[RESOURCE_INSTR_WIDTH-1:RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH];
    assign payload_0_1 = instr_0[RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0];
    assign opcode_0_3 = instr_0[RESOURCE_INSTR_WIDTH-1:RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH];
    assign payload_0_3 = instr_0[RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0];
    assign opcode_1_2 = instr_1[RESOURCE_INSTR_WIDTH-1:RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH];
    assign payload_1_2 = instr_1[RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0];
    
    dsu_t dsu_0_1, dsu_0_3, dsu_1_2;
    rep_t rep_0_1, rep_0_3, rep_1_2;
    repx_t repx_0_1, repx_0_3, repx_1_2;

    logic [IO_ADDR_WIDTH-1:0] addr_0_1;
    logic [SRAM_ADDR_WIDTH-1:0] addr_0_3, addr_1_2;
    logic dsu_valid_0_1, dsu_valid_0_3, dsu_valid_1_2;
    logic rep_valid_0_1, rep_valid_0_3, rep_valid_1_2;
    logic repx_valid_0_1, repx_valid_0_3, repx_valid_1_2;
    logic port_enable_0_1, port_enable_0_3, port_enable_1_2;
    logic [IO_ADDR_WIDTH-1:0] port_addr_0_1;
    logic [SRAM_ADDR_WIDTH-1:0] port_addr_0_3, port_addr_1_2;

    assign dsu_valid_0_1 = instr_en_0 && (opcode_0_1 == 6);
    assign rep_valid_0_1 = instr_en_0 && (opcode_0_1 == 0);
    assign repx_valid_0_1 = instr_en_0 && (opcode_0_1 == 1);
    assign dsu_0_1 = dsu_valid_0_1 ? unpack_dsu(payload_0_1) : '{default: 0};
    assign rep_0_1 = rep_valid_0_1 ? unpack_rep(payload_0_1) : '{default: 0};
    assign repx_0_1 = repx_valid_0_1 ? unpack_repx(payload_0_1) : '{default: 0};
    assign dsu_valid_0_3 = instr_en_0 && (opcode_0_3 == 6);
    assign rep_valid_0_3 = instr_en_0 && (opcode_0_3 == 0);
    assign repx_valid_0_3 = instr_en_0 && (opcode_0_3 == 1);
    assign dsu_0_3 = dsu_valid_0_3 ? unpack_dsu(payload_0_3) : '{default: 0};
    assign rep_0_3 = rep_valid_0_3 ? unpack_rep(payload_0_3) : '{default: 0};
    assign repx_0_3 = repx_valid_0_3 ? unpack_repx(payload_0_3) : '{default: 0};
    assign dsu_valid_1_2 = instr_en_1 && (opcode_1_2 == 6);
    assign rep_valid_1_2 = instr_en_1 && (opcode_1_2 == 0);
    assign repx_valid_1_2 = instr_en_1 && (opcode_1_2 == 1);
    assign dsu_1_2 = dsu_valid_1_2 ? unpack_dsu(payload_1_2) : '{default: 0};
    assign rep_1_2 = rep_valid_1_2 ? unpack_rep(payload_1_2) : '{default: 0};
    assign repx_1_2 = repx_valid_1_2 ? unpack_repx(payload_1_2) : '{default: 0};

    logic [IO_ADDR_WIDTH-1:0] step_0_1;
    logic [IO_ADDR_WIDTH-1:0] delay_0_1;
    logic [IO_ADDR_WIDTH-1:0] iter_0_1;
    logic [IO_ADDR_WIDTH-1:0] init_addr_0_1;

    logic [SRAM_ADDR_WIDTH-1:0] step_0_3, step_1_2;
    logic [SRAM_ADDR_WIDTH-1:0] delay_0_3, delay_1_2;
    logic [SRAM_ADDR_WIDTH-1:0] iter_0_3, iter_1_2;
    logic [SRAM_ADDR_WIDTH-1:0] init_addr_0_3, init_addr_1_2;

    assign step_0_1 = {10'b0, rep_0_1._step};
    assign delay_0_1 = {10'b0, rep_0_1._delay};
    assign iter_0_1 = {10'b0, rep_0_1._iter};
    assign init_addr_0_1 = dsu_0_1._init_addr;
    assign step_0_3 = rep_0_3._step[SRAM_ADDR_WIDTH-1:0];
    assign delay_0_3 = rep_0_3._delay[SRAM_ADDR_WIDTH-1:0];
    assign iter_0_3 = rep_0_3._iter[SRAM_ADDR_WIDTH-1:0];
    assign init_addr_0_3 = dsu_0_3._init_addr[SRAM_ADDR_WIDTH-1:0];
    assign step_1_2 = rep_1_2._step[SRAM_ADDR_WIDTH-1:0];
    assign delay_1_2 = rep_1_2._delay[SRAM_ADDR_WIDTH-1:0];
    assign iter_1_2 = rep_1_2._iter[SRAM_ADDR_WIDTH-1:0];
    assign init_addr_1_2 = dsu_1_2._init_addr[SRAM_ADDR_WIDTH-1:0];
    
    agu #(
      .ADDRESS_WIDTH(IO_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
    ) aug_0_0 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[1]),
      .load_initial(dsu_valid_0_1 & dsu_0_1._port == 1),
      .load_level(rep_valid_0_1 & rep_0_1._port == 1),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep_0_1._level[1:0]),
      .step(step_0_1),
      .delay(delay_0_1),
      .iterations(iter_0_1),
      .initial_address(init_addr_0_1),
      .address_valid(port_enable_0_1),
      .address(port_addr_0_1)
  );

  agu #(
      .ADDRESS_WIDTH(SRAM_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
  ) agu_0_2 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[3]),
      .load_initial(dsu_valid_0_3 & dsu_0_3._port == 3),
      .load_level(rep_valid_0_3 & rep_0_3._port == 3),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep_0_3._level[1:0]),
      .step(step_0_3),
      .delay(delay_0_3),
      .iterations(iter_0_3),
      .initial_address(init_addr_0_3),
      .address_valid(port_enable_0_3),
      .address(port_addr_0_3)
  );

  agu #(
      .ADDRESS_WIDTH(SRAM_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
  ) agu_1_2 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_1[2]),
      .load_initial(dsu_valid_1_2 & dsu_1_2._port == 2),
      .load_level(rep_valid_1_2 & rep_1_2._port == 2),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep_1_2._level[1:0]),
      .step(step_1_2),
      .delay(delay_1_2),
      .iterations(iter_1_2),
      .initial_address(init_addr_1_2),
      .address_valid(port_enable_1_2),
      .address(port_addr_1_2)
  );

    // merge the IO writing and normal writing to SRAM, IO has higher priority.
    // merge the writing and reading, writing has higher priority.
    logic [SRAM_ADDR_WIDTH-1:0] sram_in_addr;
    logic sram_in_en;
    logic [BULK_BITWIDTH-1:0] sram_in_data;
    logic [SRAM_ADDR_WIDTH-1:0] sram_out_addr;
    logic sram_out_en;
    logic [BULK_BITWIDTH-1:0] sram_out_data;
    logic [SRAM_ADDR_WIDTH-1:0] sram_addr;
    logic sram_write_en;
    always_comb begin
        if (port_enable_1_2) begin
            sram_in_en = 1;
            sram_in_addr = port_addr_1_2;
            sram_in_data = bulk_data_in_1;
        end
        else begin
            sram_in_en = 0;
            sram_in_addr = 0;
            sram_in_data = 0;
        end
    end
    always_comb begin
        if (port_enable_0_3) begin
            sram_out_en = 1;
            sram_out_addr = port_addr_0_3;
        end
        else begin
            sram_out_en = 0;
            sram_out_addr = 0;
        end
    end

    logic port_enable_0_3_delay;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            port_enable_0_3_delay <= 0;
        end
        else begin
            port_enable_0_3_delay <= port_enable_0_3;
        end
    end
    
    always_comb begin
        if (port_enable_0_3_delay) begin
            io_data_out = sram_out_data;
        end
        else begin
            io_data_out = 0;
        end
    end

    // IO
    assign io_en_out = port_enable_0_1;
    assign io_addr_out = port_addr_0_1;

    //`ifndef USE_SRAM_MODEL
  if (DEPTH == 64 && WIDTH == 256) begin : sram_64x256
    // Use the sram macro if USE_SRAM_MODEL is not defined
    sram_model sram_inst (
        .RTSEL(2'b0),
        .WTSEL(2'b0),
        .PTSEL(2'b0),
        .AA(sram_in_addr),
        .DA(sram_in_data),
        .BWEBA(1'b0),
        .WEBA(~sram_in_en),
        .CEBA(~sram_in_en),
        .CLK(clk),
        .AB(sram_out_addr),
        .DB(256'b0),
        .BWEBB(1'b0),
        .WEBB(~sram_out_en),
        .CEBB(~sram_out_en),
        .AWT(1'b0),
        .QB(sram_out_data)
    );
    // Use the sram_model if USE_SRAM_MODEL is defined
  end else begin
    $error("Unsupported DEPTH and WIDTH combination %0d x %0d", DEPTH, WIDTH);
  end

endmodule

{% endif %}