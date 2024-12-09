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

    parameter OPCODE_SWB = 3'b100;
    parameter OPCODE_ROUTE = 3'b101;

    parameter NUM_OPTIONS = 4;

endpackage

module {{fingerprint}}
import {{fingerprint}}_pkg::*;
(
    input  logic clk_0,
    input  logic rst_n_0,
    input  logic instr_en_0,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr_0,
    input  logic [FSM_PER_SLOT-1:0] activate_0,
    input logic [NUM_SLOTS-1:0][WORD_BITWIDTH-1:0] word_channels_in,
    output logic [NUM_SLOTS-1:0][WORD_BITWIDTH-1:0] word_channels_out,
    input logic [NUM_SLOTS-1:0][BULK_BITWIDTH-1:0] bulk_intracell_in,
    output logic [NUM_SLOTS-1:0][BULK_BITWIDTH-1:0] bulk_intracell_out,
    input logic [BULK_BITWIDTH-1:0] bulk_intercell_n_in,
    input logic [BULK_BITWIDTH-1:0] bulk_intercell_w_in,
    input logic [BULK_BITWIDTH-1:0] bulk_intercell_e_in,
    input logic [BULK_BITWIDTH-1:0] bulk_intercell_s_in,
    output logic [BULK_BITWIDTH-1:0] bulk_intercell_n_out,
    output logic [BULK_BITWIDTH-1:0] bulk_intercell_w_out,
    output logic [BULK_BITWIDTH-1:0] bulk_intercell_e_out,
    output logic [BULK_BITWIDTH-1:0] bulk_intercell_s_out
);

    logic clk, rst_n;
    assign clk = clk_0;
    assign rst_n = rst_n_0;

    logic [2:0] opcode;
    logic [23:0] payload;
    swb_t swb;
    route_t route;

    logic [NUM_OPTIONS-1:0][NUM_SLOTS-1:0][3:0] swb_configs, swb_configs_next;
    logic [NUM_OPTIONS-1:0][3:0] route_in_src_configs, route_in_src_configs_next;
    logic [NUM_OPTIONS-1:0][15:0] route_in_dst_configs, route_in_dst_configs_next;
    logic [NUM_OPTIONS-1:0][3:0] route_out_src_configs, route_out_src_configs_next;
    logic [NUM_OPTIONS-1:0][15:0] route_out_dst_configs, route_out_dst_configs_next;

    logic [NUM_SLOTS-1:0][3:0] curr_swb_configs, curr_swb_configs_next;
    logic [3:0] curr_route_in_src_configs, curr_route_in_src_configs_next;
    logic [15:0] curr_route_in_dst_configs, curr_route_in_dst_configs_next;
    logic [3:0] curr_route_out_src_configs, curr_route_out_src_configs_next;
    logic [15:0] curr_route_out_dst_configs, curr_route_out_dst_configs_next;

    logic [1:0] current_option;

    assign current_option = 0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            swb_configs = 0;
            route_in_src_configs = 0;
            route_in_dst_configs = 0;
            route_out_src_configs = 0;
            route_out_dst_configs = 0;
            curr_swb_configs = 0;
            curr_route_in_src_configs = 0;
            curr_route_in_dst_configs = 0;
            curr_route_out_src_configs = 0;
        end else begin
            swb_configs = swb_configs_next;
            route_in_src_configs = route_in_src_configs_next;
            route_in_dst_configs = route_in_dst_configs_next;
            route_out_src_configs = route_out_src_configs_next;
            route_out_dst_configs = route_out_dst_configs_next;
            curr_swb_configs = curr_swb_configs_next;
            curr_route_in_src_configs = curr_route_in_src_configs_next;
            curr_route_in_dst_configs = curr_route_in_dst_configs_next;
            curr_route_out_src_configs = curr_route_out_src_configs_next;
            curr_route_out_dst_configs = curr_route_out_dst_configs_next;
        end
    end

    always_comb begin
      opcode = 0;
      payload = 0;
      swb_configs_next = swb_configs;
      route_in_src_configs_next = route_in_src_configs;
      route_in_dst_configs_next = route_in_dst_configs;
      route_out_src_configs_next = route_out_src_configs;
      route_out_dst_configs_next = route_out_dst_configs;
      curr_swb_configs_next = curr_swb_configs;
      curr_route_in_src_configs_next = curr_route_in_src_configs;
      curr_route_in_dst_configs_next = curr_route_in_dst_configs;
      curr_route_out_src_configs_next = curr_route_out_src_configs;

      if (activate_0[0]) begin
        curr_swb_configs_next = swb_configs[current_option];
      end
      if (activate_0[2]) begin
        curr_route_in_src_configs_next = route_in_src_configs[current_option];
        curr_route_in_dst_configs_next = route_in_dst_configs[current_option];
        curr_route_out_src_configs_next = route_out_src_configs[current_option];
        curr_route_out_dst_configs_next = route_out_dst_configs[current_option];
      end
      if (instr_en_0) begin
          opcode = instr_0[26:24];
          payload = instr_0[23:0];
          case(opcode)
              OPCODE_SWB: begin
                      // Switchbox
                      swb = unpack_swb(payload);
                      swb_configs_next[swb._option][swb._target] = swb._source;
                  end
              OPCODE_ROUTE: begin
                      // Router
                      route = unpack_route(payload);
                      if (route._sr==1'b0) begin // send
                          route_out_src_configs_next[route._option] = route._source;
                          route_out_dst_configs_next[route._option] = route._target;
                      end else begin // receive
                          route_in_src_configs_next[route._option] = route._source;
                          route_in_dst_configs_next[route._option] = route._target;
                      end
              end
              default: begin
                      // Do nothing
              end
          endcase
      end
    end

    logic [BULK_BITWIDTH-1:0] bulk_intercell_c_in, bulk_intercell_c_out;
    logic [BULK_BITWIDTH-1:0] bulk_intercell_self;
    always_comb begin
      bulk_intercell_c_in = 0;
      bulk_intercell_c_out = 0;
      bulk_intercell_self = 0;
      bulk_intercell_e_out = 0;
      bulk_intercell_n_out = 0;
      bulk_intercell_w_out = 0;
      bulk_intercell_s_out = 0;

      for(int i=0; i<NUM_SLOTS; i=i+1) begin
        word_channels_out[i] = word_channels_in[curr_swb_configs[i]];
      end

      bulk_intercell_c_out = bulk_intracell_in[curr_route_out_src_configs];
      
      if (curr_route_out_dst_configs[1]) begin
        bulk_intercell_n_out = bulk_intercell_c_out;
      end else if (curr_route_out_dst_configs[3]) begin
        bulk_intercell_w_out = bulk_intercell_c_out;
      end else if (curr_route_out_dst_configs[3]) begin
        bulk_intercell_self = bulk_intercell_c_out;
      end else if (curr_route_out_dst_configs[5]) begin
        bulk_intercell_e_out = bulk_intercell_c_out;
      end else if (curr_route_out_dst_configs[7]) begin
        bulk_intercell_s_out = bulk_intercell_c_out;
      end

      if (curr_route_in_src_configs == 1) begin
        bulk_intercell_c_in = bulk_intercell_n_in;
      end else if (curr_route_out_src_configs == 3) begin
        bulk_intercell_c_in = bulk_intercell_w_in;
      end else if (curr_route_out_src_configs == 4) begin
        bulk_intercell_c_in = bulk_intercell_self;
      end else if (curr_route_out_src_configs == 5) begin
        bulk_intercell_c_in = bulk_intercell_e_in;
      end else if (curr_route_out_src_configs == 7) begin
        bulk_intercell_c_in = bulk_intercell_s_in;
      end

      for(int i=0; i<NUM_SLOTS; i=i+1) begin
        if(curr_route_in_dst_configs[i]) begin
          bulk_intracell_out[i] = bulk_intercell_c_in;
        end
      end

    end
endmodule

{% endif %}