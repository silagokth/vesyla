package input_output_cell_pkg;
    parameter IO_DATA_WIDTH = 256;
    parameter IO_ADDR_WIDTH = 16;
    parameter RESOURCE_INSTR_WIDTH = 27;
    parameter NUM_SLOTS = 16;
    parameter INSTR_DATA_WIDTH = 32;
    parameter INSTR_ADDR_WIDTH = 6;
    parameter INSTR_HOPS_WIDTH = 4;
endpackage

module input_output_cell
import input_output_cell_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    input  logic call_in,
    output logic call_out,
    input  logic ret_in,
    output logic ret_out,
    output logic io_en_in,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_in,
    input  logic [IO_DATA_WIDTH-1:0] io_data_in,
    output logic io_en_out,
    output logic [IO_ADDR_WIDTH-1:0] io_addr_out,
    output logic [IO_DATA_WIDTH-1:0] io_data_out,
    input  logic [INSTR_DATA_WIDTH-1:0] instr_data_in,
    input  logic [INSTR_ADDR_WIDTH-1:0] instr_addr_in,
    input  logic [INSTR_HOPS_WIDTH-1:0] instr_hops_in,
    input  logic instr_en_in,
    output logic [INSTR_DATA_WIDTH-1:0] instr_data_out,
    output logic [INSTR_ADDR_WIDTH-1:0] instr_addr_out,
    output logic [INSTR_HOPS_WIDTH-1:0] instr_hops_out,
    output logic instr_en_out
);

    logic [NUM_SLOTS-1:0] instr_en;
    logic [RESOURCE_INSTR_WIDTH-1:0] instr;
    logic [NUM_SLOTS-1:0] activate;

    logic controller_call;
    logic controller_ret, controller_ret_remember, ret_in_remember;
    assign controller_call = call_in;
    always_ff @(negedge rst_n or posedge clk)
    begin
        if (~rst_n) begin
            call_out <= 0;
            controller_ret_remember <= 0;
            ret_in_remember <= 0;
        end else begin
            if (controller_ret ) begin
                controller_ret_remember <= 1;
            end else if (ret_in) begin
                ret_in_remember <= 1;
            end
            call_out <= call_in;
        end
    end

    assign ret_out = controller_ret_remember & ret_in_remember;

    sequencer sequencer_inst
    (
        .clk(clk),
        .rst_n(rst_n),
        .call(controller_call),
        .ret(controller_ret),
        .instr_en(instr_en),
        .instr(instr),
        .activate(activate),
        .instr_data_in(instr_data_in),
        .instr_addr_in(instr_addr_in),
        .instr_hops_in(instr_hops_in),
        .instr_en_in(instr_en_in),
        .instr_data_out(instr_data_out),
        .instr_addr_out(instr_addr_out),
        .instr_hops_out(instr_hops_out),
        .instr_en_out(instr_en_out)
    );

    dummy dummy_inst
    (
        .clk(clk),
        .rst_n(rst_n),
        .instr_en(instr_en[0]),
        .instr(instr),
        .activate(activate[0])
    );

    vec_add vec_add_inst
    (
        .clk(clk),
        .rst_n(rst_n),
        .instr_en(instr_en[1]),
        .instr(instr),
        .activate(activate[1]),
        .io_en_in(io_en_in),
        .io_addr_in(io_addr_in),
        .io_data_in(io_data_in),
        .io_en_out(io_en_out),
        .io_addr_out(io_addr_out),
        .io_data_out(io_data_out)
    );

endmodule