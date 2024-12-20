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
    output logic io_en_in,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_in,
    input  logic [IO_DATA_WIDTH-1:0] io_data_in
);

    // Useless outputs:
    assign word_data_out_0 = 0;
    assign word_data_out_1 = 0;
    assign word_data_out_2 = 0;
    assign word_data_out_3 = 0;
    assign bulk_data_out_0 = 0;
    assign bulk_data_out_2 = 0;
    assign bulk_data_out_3 = 0;

    logic clk, rst_n;
    assign clk = clk_0;
    assign rst_n = rst_n_0;

    logic [INSTR_OPCODE_BITWIDTH-1:0] opcode_0_0, opcode_0_2, opcode_1_3;
    logic [RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0] payload_0_0, payload_0_2, payload_1_3;

    assign opcode_0_0 = instr_0[RESOURCE_INSTR_WIDTH-1:RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH];
    assign payload_0_0 = instr_0[RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0];
    assign opcode_0_2 = instr_0[RESOURCE_INSTR_WIDTH-1:RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH];
    assign payload_0_2 = instr_0[RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0];
    assign opcode_1_3 = instr_1[RESOURCE_INSTR_WIDTH-1:RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH];
    assign payload_1_3 = instr_1[RESOURCE_INSTR_WIDTH-INSTR_OPCODE_BITWIDTH-1:0];
    
    dsu_t dsu_0_0, dsu_0_2, dsu_1_3;
    rep_t rep_0_0, rep_0_2, rep_1_3;
    repx_t repx_0_0, repx_0_2, repx_1_3;

    logic [IO_ADDR_WIDTH-1:0] addr_0_0;
    logic [SRAM_ADDR_WIDTH-1:0] addr_0_2, addr_1_3;
    logic dsu_valid_0_0, dsu_valid_0_2, dsu_valid_1_3;
    logic rep_valid_0_0, rep_valid_0_2, rep_valid_1_3;
    logic repx_valid_0_0, repx_valid_0_2, repx_valid_1_3;
    logic port_enable_0_0, port_enable_0_2, port_enable_1_3;
    logic [IO_ADDR_WIDTH-1:0] port_addr_0_0;
    logic [SRAM_ADDR_WIDTH-1:0] port_addr_0_2, port_addr_1_3;

    assign dsu_valid_0_0 = instr_en_0 && (opcode_0_0 == 6);
    assign rep_valid_0_0 = instr_en_0 && (opcode_0_0 == 0);
    assign repx_valid_0_0 = instr_en_0 && (opcode_0_0 == 1);
    assign dsu_0_0 = dsu_valid_0_0 ? unpack_dsu(payload_0_0) : '{default: 0};
    assign rep_0_0 = rep_valid_0_0 ? unpack_rep(payload_0_0) : '{default: 0};
    assign repx_0_0 = repx_valid_0_0 ? unpack_repx(payload_0_0) : '{default: 0};
    assign dsu_valid_0_2 = instr_en_0 && (opcode_0_2 == 6);
    assign rep_valid_0_2 = instr_en_0 && (opcode_0_2 == 0);
    assign repx_valid_0_2 = instr_en_0 && (opcode_0_2 == 1);
    assign dsu_0_2 = dsu_valid_0_2 ? unpack_dsu(payload_0_2) : '{default: 0};
    assign rep_0_2 = rep_valid_0_2 ? unpack_rep(payload_0_2) : '{default: 0};
    assign repx_0_2 = repx_valid_0_2 ? unpack_repx(payload_0_2) : '{default: 0};
    assign dsu_valid_1_3 = instr_en_1 && (opcode_1_3 == 6);
    assign rep_valid_1_3 = instr_en_1 && (opcode_1_3 == 0);
    assign repx_valid_1_3 = instr_en_1 && (opcode_1_3 == 1);
    assign dsu_1_3 = dsu_valid_1_3 ? unpack_dsu(payload_1_3) : '{default: 0};
    assign rep_1_3 = rep_valid_1_3 ? unpack_rep(payload_1_3) : '{default: 0};
    assign repx_1_3 = repx_valid_1_3 ? unpack_repx(payload_1_3) : '{default: 0};

    logic [IO_ADDR_WIDTH-1:0] step_0_0;
    logic [IO_ADDR_WIDTH-1:0] delay_0_0;
    logic [IO_ADDR_WIDTH-1:0] iter_0_0;
    logic [IO_ADDR_WIDTH-1:0] init_addr_0_0;

    logic [SRAM_ADDR_WIDTH-1:0] step_0_2, step_1_3;
    logic [SRAM_ADDR_WIDTH-1:0] delay_0_2, delay_1_3;
    logic [SRAM_ADDR_WIDTH-1:0] iter_0_2, iter_1_3;
    logic [SRAM_ADDR_WIDTH-1:0] init_addr_0_2, init_addr_1_3;

    assign step_0_0 = {10'b0, rep_0_0._step};
    assign delay_0_0 = {10'b0, rep_0_0._delay};
    assign iter_0_0 = {10'b0, rep_0_0._iter};
    assign init_addr_0_0 = dsu_0_0._init_addr;
    assign step_0_2 = rep_0_2._step[SRAM_ADDR_WIDTH-1:0];
    assign delay_0_2 = rep_0_2._delay[SRAM_ADDR_WIDTH-1:0];
    assign iter_0_2 = rep_0_2._iter[SRAM_ADDR_WIDTH-1:0];
    assign init_addr_0_2 = dsu_0_2._init_addr[SRAM_ADDR_WIDTH-1:0];
    assign step_1_3 = rep_1_3._step[SRAM_ADDR_WIDTH-1:0];
    assign delay_1_3 = rep_1_3._delay[SRAM_ADDR_WIDTH-1:0];
    assign iter_1_3 = rep_1_3._iter[SRAM_ADDR_WIDTH-1:0];
    assign init_addr_1_3 = dsu_1_3._init_addr[SRAM_ADDR_WIDTH-1:0];
    
    agu #(
      .ADDRESS_WIDTH(IO_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
    ) aug_0_0 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[0]),
      .load_initial(dsu_valid_0_0 & dsu_0_0._port == 0),
      .load_level(rep_valid_0_0 & rep_0_0._port == 0),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep_0_0._level[1:0]),
      .step(step_0_0),
      .delay(delay_0_0),
      .iterations(iter_0_0),
      .initial_address(init_addr_0_0),
      .address_valid(port_enable_0_0),
      .address(port_addr_0_0)
  );

  agu #(
      .ADDRESS_WIDTH(SRAM_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
  ) agu_0_2 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_0[2]),
      .load_initial(dsu_valid_0_2 & dsu_0_2._port == 2),
      .load_level(rep_valid_0_2 & rep_0_2._port == 2),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep_0_2._level[1:0]),
      .step(step_0_2),
      .delay(delay_0_2),
      .iterations(iter_0_2),
      .initial_address(init_addr_0_2),
      .address_valid(port_enable_0_2),
      .address(port_addr_0_2)
  );

  agu #(
      .ADDRESS_WIDTH(SRAM_ADDR_WIDTH),
      .NUMBER_OF_LEVELS(4)
  ) agu_1_3 (
      .clk(clk),
      .rst_n(rst_n),
      .activate(activate_1[3]),
      .load_initial(dsu_valid_1_3 & dsu_1_3._port == 3),
      .load_level(rep_valid_1_3 & rep_1_3._port == 3),
      .is_extended(),  // TODO: not supported yet
      .level_to_load(rep_1_3._level[1:0]),
      .step(step_1_3),
      .delay(delay_1_3),
      .iterations(iter_1_3),
      .initial_address(init_addr_1_3),
      .address_valid(port_enable_1_3),
      .address(port_addr_1_3)
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
        if (port_enable_0_2) begin
            sram_in_en = 1;
            sram_in_addr = port_addr_0_2;
            sram_in_data = io_data_in;
        end
        else begin
            sram_in_en = 0;
            sram_in_addr = 0;
            sram_in_data = 0;
        end
    end
    always_comb begin
        if (port_enable_1_3) begin
            sram_out_en = 1;
            sram_out_addr = port_addr_1_3;
        end
        else begin
            sram_out_en = 0;
            sram_out_addr = 0;
        end
    end

    logic port_enable_1_3_delay;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            port_enable_1_3_delay <= 0;
        end
        else begin
            port_enable_1_3_delay <= port_enable_1_3;
        end
    end
    
    always_comb begin
        if (port_enable_1_3_delay) begin
            bulk_data_out_1 = sram_out_data;
        end
        else begin
            bulk_data_out_1 = 0;
        end
    end

    // IO
    assign io_en_in = port_enable_0_0;
    assign io_addr_in = port_addr_0_0;

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




module agu #(
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


  logic [NUMBER_OF_LEVELS-1:0] count_states;
  logic [NUMBER_OF_LEVELS+1:0] level_finish;
  logic [NUMBER_OF_LEVELS:0][ADDRESS_WIDTH-1:0] addresses;
  logic [NUMBER_OF_LEVELS:0] level_restart;


  // initial level
  agu_initial #(
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
    agu_level #(
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
        .lower_level_finish(level_finish[i]),
        .higher_level_finish(level_finish[i+2]),
        .higher_level_restart(level_restart[i+1]),
        .count_state(count_states[i]),
        .finish(level_finish[i+1]),
        .address(addresses[i+1]),
        .lower_level_restart(level_restart[i])
    );
  end

  assign level_finish[0] = 1;
  assign level_finish[NUMBER_OF_LEVELS+1] = 1;
  assign level_restart[NUMBER_OF_LEVELS] = 1;

  // adder tree to calculate the address
  always_comb begin
    address = addresses[0];
    for (int i = 1; i <= NUMBER_OF_LEVELS; i++) begin
      address = address + addresses[i];
    end
  end

  assign address_valid = |count_states;

endmodule

module agu_initial #(
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

module agu_level #(
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
    input logic lower_level_finish,
    input logic higher_level_finish,
    input logic higher_level_restart,
    output logic count_state,
    output logic finish,
    output logic [ADDR_WIDTH-1:0] address,
    output logic lower_level_restart
);

  typedef enum logic [2:0] {
    IDLE,
    WAIT,
    DELAY,
    COUNT,
    LOAD
  } state_t;

  state_t state, next_state;
  logic [ADDR_WIDTH-1:0] next_address;
  logic [ADDR_WIDTH-1:0] delay_reg;
  logic [ADDR_WIDTH-1:0] delay_counter;
  logic [ADDR_WIDTH-1:0] delay_counter_next;
  logic [ADDR_WIDTH-1:0] step_reg;
  logic [ADDR_WIDTH-1:0] iterations_reg;
  logic [ADDR_WIDTH-1:0] iter_counter;
  logic [ADDR_WIDTH-1:0] iter_counter_next;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= IDLE;
      address <= 0;
    end else begin
      state   <= next_state;
      address <= next_address;
    end
  end

  assign finish = (iter_counter == iterations_reg && lower_level_finish);
  assign count_state  = (state == COUNT);

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
      iter_counter <= 0;
    end else begin
      delay_counter <= delay_counter_next;
      iter_counter <= iter_counter_next;
    end
  end



  always_comb begin
    next_state = state;
    next_address = address;
    delay_counter_next = delay_counter;
    iter_counter_next = iter_counter;
    lower_level_restart = 0;

    case (state)
      IDLE: begin
        if (activate) begin
          next_state = COUNT;
          next_address = 0;
          iter_counter_next = 0;
          delay_counter_next = 0;
        end else begin
          next_state   = IDLE;
          next_address = 0;
          delay_counter_next=0;
          iter_counter_next=0;
        end
      end

      COUNT: begin
        if (lower_level_finish) begin
            if (higher_level_finish) begin
              if (finish) begin
                next_state = IDLE;
                next_address = 0;
                iter_counter_next = 0;
              end else begin
                if (delay_reg>0) begin
                  next_state   = DELAY;
                  next_address = address;
                  iter_counter_next = iter_counter;
                end else begin
                  next_state   = COUNT;
                  next_address = address + step_reg;
                  iter_counter_next = iter_counter + 1;
                  lower_level_restart = 1;
                end
              end
            end else begin
              if (finish) begin
                if (higher_level_restart) begin
                  next_state = COUNT;
                  next_address = 0;
                  iter_counter_next = 0;
                  lower_level_restart = 1;
                end else begin
                  next_state = WAIT;
                  next_address = address;
                  iter_counter_next = iter_counter;
                end

              end else begin
                if (delay_reg>0) begin
                  next_state   = DELAY;
                  next_address = address;
                  iter_counter_next = iter_counter;
                end else begin
                  next_state   = COUNT;
                  next_address = address + step_reg;
                  iter_counter_next = iter_counter + 1;
                  lower_level_restart = 1;
                end
              end
            end
        end else begin
          next_state = WAIT;
          next_address = address;
          iter_counter_next = iter_counter;
          delay_counter_next = 0;
        end
      end

      WAIT: begin
        if (lower_level_finish) begin
          if (higher_level_finish) begin
            if (finish) begin
              next_state = IDLE;
              next_address = 0;
              iter_counter_next = 0;
            end else begin
              if (delay_reg>0) begin
                next_state   = DELAY;
                next_address = address;
                iter_counter_next = iter_counter;
              end else begin
                next_state   = COUNT;
                next_address = address + step_reg;
                iter_counter_next = iter_counter + 1;
                lower_level_restart = 1;
              end
            end
          end else begin
            if (finish) begin
              if (higher_level_restart) begin
                next_state = COUNT;
                next_address = 0;
                iter_counter_next = 0;
                lower_level_restart = 1;
              end else begin
                next_state = WAIT;
                next_address = address;
                iter_counter_next = iter_counter;
              end
            end else begin
              if (delay_reg>0) begin
                next_state   = DELAY;
                next_address = address;
                iter_counter_next = iter_counter;
              end else begin
                next_state   = COUNT;
                next_address = address + step_reg;
                iter_counter_next = iter_counter + 1;
                lower_level_restart = 1;
              end
            end
          end
        end else begin
          next_state   = WAIT;
          next_address = address;
        end
      end
      DELAY: begin
        delay_counter_next = delay_counter + 1;
        if (delay_counter_next == delay_reg) begin
          next_state   = COUNT;
          next_address = address+step_reg;
          iter_counter_next = iter_counter+1;
          lower_level_restart = 1;
          delay_counter_next = 0;
        end else begin
          next_state   = DELAY;
          next_address = address;
          iter_counter_next = iter_counter;
        end
      end
      default: begin
        next_state   = IDLE;
        next_address = 0;
        iter_counter_next = 0;
        delay_counter_next = 0;
      end
    endcase
  end

endmodule


module sram_model #(
    parameter ADDR_WIDTH = 6,
    parameter DATA_WIDTH = 256
)
    (
    input logic [1:0] RTSEL,
    input logic [1:0] WTSEL,
    input logic [1:0] PTSEL,
    input logic [ADDR_WIDTH-1:0] AA,
    input logic [DATA_WIDTH-1:0] DA,
    input logic BWEBA,
    input logic WEBA,CEBA,CLK,
    input logic [ADDR_WIDTH-1:0] AB,
    input logic [DATA_WIDTH-1:0] DB,
    input logic BWEBB,
    input logic WEBB,CEBB,
    input logic AWT,
    output logic [DATA_WIDTH-1:0] QA,
    output logic [DATA_WIDTH-1:0] QB
  );

    localparam numWord = 2 ** ADDR_WIDTH;

  //=== Internal Memory Declaration ===//
  reg [numWord-1:0][DATA_WIDTH-1:0] memory;

  //=== SRAM Functionality ===//
  always @(posedge CLK) begin
      QA <= 0;
      if (!CEBA) begin
        memory[AA] <= DA;
      end else begin
        memory[AA] <= memory[AA];
      end
      if (!CEBB) begin
        QB <= memory[AB];
      end else begin
        QB <= 0;
      end
  end
endmodule

{% endif %}