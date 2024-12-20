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

  parameter FSM_MAX_STATES = 4;
  parameter FSM_DELAY_WIDTH = 3;
  parameter DPU_MODE_WIDTH = 6;
  parameter DPU_IMMEDIATE_WIDTH = 8;
  parameter OPCODE_WIDTH = 4;
  parameter BITWIDTH = 16;
  parameter INSTRUCTION_PAYLOAD_WIDTH = 27;
  parameter OPCODE_H = 26;
  parameter OPCODE_L = 24;
  parameter OPCODE_DPU = 3;
  parameter OPCODE_FSM = 2;
  parameter DPU_MODE_ADD = 1;
  parameter DPU_MODE_MUL = 7;
  parameter DPU_MODE_MAC = 10;

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
    output logic [BULK_BITWIDTH-1:0] bulk_data_out_1
);

    logic clk, rst_n, instruction_valid, activate;
    logic [INSTRUCTION_PAYLOAD_WIDTH-1:0] instruction;
    assign clk = clk_0;
    assign rst_n = rst_n_0;
    assign instruction_valid = instr_en_0;
    assign activate = activate_0[0];
    assign instruction = instr_0;

    // useless output
    assign word_data_out_1 = 0;
    assign bulk_data_out_0 = 0;
    assign bulk_data_out_1 = 0;

    // register inputs
    logic [WORD_BITWIDTH-1:0] in0, in1, out0;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in0 <= 0;
            in1 <= 0;
        end else begin
            in0 <= word_data_in_0;
            in1 <= word_data_in_1;
        end
    end

  logic [$clog2(FSM_MAX_STATES)-1:0][DPU_MODE_WIDTH-1:0] mode_memory;
  logic [$clog2(FSM_MAX_STATES)-1:0][DPU_IMMEDIATE_WIDTH-1:0] immediate_memory;
  logic [$clog2(FSM_MAX_STATES)-1:0] fsm_option;
  logic [DPU_IMMEDIATE_WIDTH-1:0] immediate;
  logic [DPU_MODE_WIDTH-1:0] mode;
  logic [OPCODE_WIDTH-1:0] opcode;
  logic dpu_valid;
  logic fsm_valid;
  logic reset_accumulator;

  dpu_t dpu;
  fsm_t fsm;

  fsm u_dpu_fsm (
      .clk(clk),
      .rst_n(rst_n),
      .activate(|activate),
      .instruction_valid(instruction_valid),
      .instruction(instruction),
      .state(fsm_option)
  );

  assign opcode = instruction[OPCODE_H:OPCODE_L];
  assign dpu_valid = instruction_valid && (opcode == OPCODE_DPU);
  assign fsm_valid = instruction_valid && (opcode == OPCODE_FSM);
  assign dpu = dpu_valid ? unpack_dpu(
          instruction[INSTRUCTION_PAYLOAD_WIDTH-1:0]
      ) :
      '{default: 0};
  assign fsm = fsm_valid ? unpack_fsm(
          instruction[INSTRUCTION_PAYLOAD_WIDTH-1:0]
      ) :
      '{default: 0};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      immediate_memory <= '0;
    end else begin
      if (dpu_valid) begin
        mode_memory[dpu._option] <= dpu._mode;
        immediate_memory[dpu._option] <= dpu._immediate;
      end
    end
  end

  assign mode = mode_memory[fsm_option];
  assign immediate = immediate_memory[fsm_option];

  logic signed [BITWIDTH-1:0] acc0, acc0_next;
  logic signed [BITWIDTH-1:0] adder_in0;
  logic signed [BITWIDTH-1:0] adder_in1;
  logic signed [BITWIDTH-1:0] adder_out;
  logic signed [BITWIDTH-1:0] mult_in0;
  logic signed [BITWIDTH-1:0] mult_in1;
  logic signed [BITWIDTH-1:0] mult_out;

  // signed saturateion
  localparam logic signed [BITWIDTH-1:0] MAX_RESULT = 2 ** (BITWIDTH - 1) - 1;
  localparam logic signed [BITWIDTH-1:0] MIN_RESULT = -2 ** (BITWIDTH - 1);


  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc0 <= '0;
    end else begin
      if (reset_accumulator) begin
        acc0 <= '0;
      end else begin
        acc0 <= acc0_next;
      end
    end
  end

  always_comb begin
    out0 = 0;
    adder_in0 = 0;
    adder_in1 = 0;
    mult_in0 = 0;
    mult_in1 = 0;

    case (mode)
      DPU_MODE_ADD: begin
        adder_in0 = in0;
        adder_in1 = in1;
        out0 = adder_out;
      end
      DPU_MODE_MAC: begin
        mult_in0 = in0;
        mult_in1 = in1;
        adder_in0 = mult_out;
        adder_in1 = acc0;
        acc0_next = adder_out;
        out0 = adder_out;
      end
      DPU_MODE_MUL: begin
        mult_in0 = in0;
        mult_in1 = in1;
        out0 = mult_out;
      end
    endcase
  end

  // in future we will use ChipWare blocks for pipelined and such 
  {{fingerprint}}_adder adder_inst (
      .in1(adder_in0),
      .in2(adder_in1),
      .saturate(1'b1),
      .out(adder_out)
  );

  {{fingerprint}}_multiplier mult_inst (
      .in1(mult_in0),
      .in2(mult_in1),
      .saturate(1'b1),
      .out(mult_out)
  );

  assign word_data_out_0 = out0;

endmodule

module {{fingerprint}}_adder
import {{fingerprint}}_pkg::*;
(
    input logic signed [BITWIDTH-1:0] in1,
    input logic signed [BITWIDTH-1:0] in2,
    output logic signed [BITWIDTH-1:0] out,
    output logic overflow,
    output logic underflow,
    input logic saturate
);

  localparam logic signed [BITWIDTH-1:0] MAX_RESULT = (1 << (BITWIDTH - 1)) - 1;
  localparam logic signed [BITWIDTH-1:0] MIN_RESULT = -MAX_RESULT - 1;
  logic signed [BITWIDTH:0] temp_add;

  always_comb begin
    overflow = 0;
    underflow = 0;
    out = 0;
    temp_add = in1 + in2;
    // saturation logic
    if (saturate) begin
      if (temp_add > MAX_RESULT) begin
        out = MAX_RESULT;
        overflow = 1;
      end else if (temp_add < MIN_RESULT) begin
        out = MIN_RESULT;
        underflow = 1;
      end else begin
        out = temp_add[BITWIDTH-1:0];
      end
    end else begin
      out = temp_add[BITWIDTH-1:0];
    end
  end
endmodule

module {{fingerprint}}_multiplier
import {{fingerprint}}_pkg::*;
(
    input logic signed [BITWIDTH-1:0] in1,
    input logic signed [BITWIDTH-1:0] in2,
    output logic signed [BITWIDTH-1:0] out,
    output logic overflow,
    output logic underflow,
    input logic saturate
);

  localparam logic signed [BITWIDTH-1:0] MAX_RESULT = (1 << (BITWIDTH - 1)) - 1;
  localparam logic signed [BITWIDTH-1:0] MIN_RESULT = -MAX_RESULT - 1;
  logic signed [BITWIDTH*2-1:0] temp_mult;

  always_comb begin
    out = 0;
    overflow = 0;
    underflow = 0;
    temp_mult = in1 * in2;
    // saturation logic
    if (saturate) begin
      if (temp_mult > MAX_RESULT) begin
        out = MAX_RESULT;
        overflow = 1;
      end else if (temp_mult < MIN_RESULT) begin
        out = MIN_RESULT;
        underflow = 1;
      end else begin
        out = temp_mult[BITWIDTH-1:0];
      end
    end else begin
      out = temp_mult[BITWIDTH-1:0];
    end
  end
endmodule

module fsm
import {{fingerprint}}_pkg::*;
(
    input logic clk,
    input logic rst_n,
    input logic activate,
    input logic instruction_valid,
    input logic [INSTRUCTION_PAYLOAD_WIDTH-1:0] instruction,
    output logic [$clog2(FSM_MAX_STATES)-1:0] state
);

  logic [INSTRUCTION_PAYLOAD_WIDTH-1:0] instruction_reg;
  logic [OPCODE_WIDTH-1:0] opcode;
  logic fsm_valid;
  logic [$clog2(FSM_MAX_STATES)-1:0] next_state;
  logic [FSM_MAX_STATES-2:0][FSM_DELAY_WIDTH-1:0] fsm_delays;
  logic [FSM_DELAY_WIDTH-1:0] delay_counter;
  logic [FSM_DELAY_WIDTH-1:0] delay_counter_next;

  fsm_t fsm;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      instruction_reg <= '0;
    end else begin
      if (instruction_valid) begin
        instruction_reg <= instruction;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      delay_counter <= fsm_delays[0];
    end else begin
      delay_counter <= delay_counter_next;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fsm_delays = '0;
    end else begin
      fsm_delays[0] = fsm._delay_0;
      fsm_delays[1] = fsm._delay_1;
      fsm_delays[2] = fsm._delay_2;

    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= 0;
    end else begin
      state <= next_state;
    end
  end

  always_comb begin
    next_state = 0;
    delay_counter_next = 0;
    if (delay_counter == 0) begin
      if (state == FSM_MAX_STATES - 1) begin
        next_state = 0;
      end else begin
        next_state = state + 1;
        delay_counter_next = fsm_delays[state];
      end
    end else begin
      delay_counter_next = delay_counter - 1;
    end
  end

  assign opcode = instruction_reg[OPCODE_H:OPCODE_L];
  assign fsm_valid = instruction_valid && (opcode == OPCODE_FSM);
  assign fsm = fsm_valid ?
      '{default: 0}
      : unpack_fsm(
          instruction_reg[INSTRUCTION_PAYLOAD_WIDTH-1:0]
      );


endmodule

{% endif %}