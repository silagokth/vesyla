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
    
    {{fingerprint}}_agu #(
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

    {{fingerprint}}_agu #(
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

  {{fingerprint}}_agu #(
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

  {{fingerprint}}_agu #(
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




module {{fingerprint}}_agu #(
    parameter ADDRESS_WIDTH = 8,
    parameter NUMBER_OF_LEVELS = 4
) (
    input logic clk,
    input logic rst_n,
    input logic activate,
    input logic load_initial,
    input logic load_level,
    input logic is_extended,
    input logic [$clog2(NUMBER_OF_LEVELS)-1:0] level_to_load,
    input logic [ADDRESS_WIDTH-1:0] step,
    input logic [ADDRESS_WIDTH-1:0] delay,
    input logic [ADDRESS_WIDTH-1:0] iterations,
    input logic [ADDRESS_WIDTH-1:0] initial_address,
    output logic address_valid,
    output logic [ADDRESS_WIDTH-1:0] address
);


  logic [NUMBER_OF_LEVELS-1:0] wait_states;
  logic [NUMBER_OF_LEVELS-1:0] delay_states;
  logic [NUMBER_OF_LEVELS-1:0] count_states;
  logic [NUMBER_OF_LEVELS-1:0] higher_levels_activations;
  logic [NUMBER_OF_LEVELS-1:0] lower_levels_activations;
  logic [NUMBER_OF_LEVELS:0][ADDRESS_WIDTH-1:0] addresses;


  // initial level
  {{fingerprint}}_agu_initial #(
      .ADDR_WIDTH(ADDRESS_WIDTH),
      .DATA_WIDTH(ADDRESS_WIDTH)
  ) agu_initial_inst (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate),
      .load_config(load_initial),
      .is_extended(is_extended),
      .initial_address(initial_address),
      .address(addresses[0])
  );

  genvar i;
  for (i = 0; i < NUMBER_OF_LEVELS; i++) begin : agu_levels
    {{fingerprint}}_agu_level #(
        .ADDR_WIDTH(ADDRESS_WIDTH),
        .DATA_WIDTH(ADDRESS_WIDTH)
    ) agu_level_inst (
        .clk(clk),
        .rst_n(rst_n),
        .activate(activate),
        .load_config(load_level && (level_to_load == i)),
        .is_extended(is_extended),
        .step(step),
        .delay(delay),
        .iterations(iterations),
        .higher_levels_activation(higher_levels_activations[i]),
        .lower_levels_activation(lower_levels_activations[i]),
        .wait_state(wait_states[i]),
        .delay_state(delay_states[i]),
        .count_state(count_states[i]),
        .address(addresses[i+1])
    );

    // higher level activation is the AND of all higher levels not counting or delay
    assign higher_levels_activations[i] = &wait_states[i:0] && &delay_states[i:0];
    // lower level activation is the AND of all lower levels not counting or delay
    assign lower_levels_activations[i] = &wait_states[NUMBER_OF_LEVELS-1:i] && &delay_states[NUMBER_OF_LEVELS-1:i];
  end

  // adder tree to calculate the address
  always_comb begin
    address = addresses[0];
    for (int i = 1; i <= NUMBER_OF_LEVELS; i++) begin
      address = address + addresses[i];
    end
  end

  assign address_valid = |count_states;

endmodule

module {{fingerprint}}_agu_initial #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 8
) (
    input logic clk,
    input logic rst_n,
    input logic activate,
    input logic load_config,
    input logic is_extended,
    input logic [ADDR_WIDTH-1:0] initial_address,
    output logic [ADDR_WIDTH-1:0] address
);

  logic [ADDR_WIDTH-1:0] initial_address_reg;

  // config loading
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      initial_address_reg <= 0;
    end else begin
      if (load_config) begin
        if (is_extended) begin
          // TODO: implement loading MSB of config values
          assert (0)
          else $fatal("AGU: Extended config not implemented yet");
        end else begin
          initial_address_reg <= initial_address;
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      address <= 0;
    end else begin
      if (activate) begin
        address <= initial_address_reg;
      end
    end
  end

endmodule

module {{fingerprint}}_agu_level #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 8
) (
    input logic clk,
    input logic rst_n,
    input logic activate,
    input logic load_config,
    input logic is_extended,
    input logic [ADDR_WIDTH-1:0] step,
    input logic [ADDR_WIDTH-1:0] delay,
    input logic [ADDR_WIDTH-1:0] iterations,
    input logic higher_levels_activation,  // AND of all higher levels not counting or delay
    input logic lower_levels_activation,  // AND of all lower levels not counting or delay
    output logic wait_state,
    output logic delay_state,
    output logic count_state,
    output logic [ADDR_WIDTH-1:0] address
);

  typedef enum logic [2:0] {
    IDLE,
    COUNT,
    DELAY,
    WAIT,
    LOAD
  } state_t;

  state_t state, next_state;
  logic [ADDR_WIDTH-1:0] next_address;
  logic [ADDR_WIDTH-1:0] delay_reg;
  logic [ADDR_WIDTH-1:0] delay_counter;
  logic [ADDR_WIDTH-1:0] delay_counter_next;
  logic [ADDR_WIDTH-1:0] step_reg;
  logic [ADDR_WIDTH-1:0] iterations_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= IDLE;
      address <= 0;
    end else begin
      state   <= next_state;
      address <= next_address;
    end
  end

  assign wait_state  = (state == WAIT);
  assign delay_state = (state == DELAY);
  assign count_state = (state == COUNT);

  // config loading registers
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      delay_reg <= 0;
      step_reg <= 0;
      iterations_reg <= 0;
    end else begin
      if (load_config) begin
        if (is_extended) begin
          // TODO: implement loading MSB of config values
          assert (0)
          else $fatal("AGU: Extended config not implemented yet");
        end else begin
          delay_reg <= delay;
          step_reg <= step;
          iterations_reg <= iterations;
        end
      end
    end
  end

  // delay counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      delay_counter <= 0;
    end else begin
      delay_counter <= delay_counter_next;
    end
  end

  always_comb begin
    next_state = state;
    next_address = address;
    delay_counter_next = delay_counter;

    case (state)
      IDLE: begin
        if (activate) begin
          next_state   = COUNT;
          next_address = 0;
        end
      end
      COUNT: begin
        if (delay_reg > 0) begin
          next_state   = DELAY;
          next_address = address;
        end else if (address == iterations_reg) begin
          next_state   = WAIT;
          next_address = address;
        end else begin
          next_state   = COUNT;
          next_address = address + step_reg;
        end
      end
      DELAY: begin
        delay_counter_next = delay_counter + 1;
        if (delay_counter_next == delay_reg) begin
          next_state   = COUNT;
          next_address = address;
        end else begin
          next_state   = DELAY;
          next_address = address;
        end
      end
      WAIT: begin
        if (higher_levels_activation) begin
          next_state   = COUNT;
          next_address = 0;
        end
      end
      default: begin
        next_state   = IDLE;
        next_address = 0;
      end
    endcase
  end

endmodule

{% endif %}