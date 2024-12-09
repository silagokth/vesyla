`define {{name}} {{fingerprint}}
`define {{name}}_pkg {{fingerprint}}_pkg

{% if not already_defined %}
package {{fingerprint}}_pkg;
    {% for p in parameters %}
    parameter {{p}} = {{parameters[p]}};
    {% endfor %}
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
endmodule
{% endif %}
