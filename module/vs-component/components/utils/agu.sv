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