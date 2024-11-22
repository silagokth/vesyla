
    name pools: []
        `define cell_impl _pv8b47jqufe
        `define cell_impl_pkg _pv8b47jqufe_pkg
    name pools: []
        
    name pools: ["cell_impl"]
    
        `define sequencer_impl _j91xtxoqyjk
        `define sequencer_impl_pkg _j91xtxoqyjk_pkg
    name pools: ["cell_impl"]
        
    name pools: ["cell_impl", "sequencer_impl"]
    
            rs.name : dummy_impl
            `define dummy_impl _fsahpf8ao5p
            `define dummy_impl_pkg _fsahpf8ao5p_pkg
    name pools: ["cell_impl", "sequencer_impl"]
        
    name pools: ["cell_impl", "sequencer_impl", "dummy_impl"]
        
    
            rs.name : vec_add_impl
            `define vec_add_impl _9tjr2rhr8kq
            `define vec_add_impl_pkg _9tjr2rhr8kq_pkg
    name pools: ["cell_impl", "sequencer_impl", "dummy_impl"]
        
    name pools: ["cell_impl", "sequencer_impl", "dummy_impl", "vec_add_impl"]
        
    
    
    name pools: ["cell_impl", "sequencer_impl"]
        `define cell_impl _pv8b47jqufe
        `define cell_impl_pkg _pv8b47jqufe_pkg
    name pools: ["cell_impl", "sequencer_impl"]
        
    name pools: ["cell_impl", "sequencer_impl", "cell_impl"]
    
        `define sequencer_impl _j91xtxoqyjk
        `define sequencer_impl_pkg _j91xtxoqyjk_pkg
    name pools: ["cell_impl", "sequencer_impl", "cell_impl"]
        
    name pools: ["cell_impl", "sequencer_impl", "cell_impl", "sequencer_impl"]
    
            rs.name : dummy_impl
            `define dummy_impl _fsahpf8ao5p
            `define dummy_impl_pkg _fsahpf8ao5p_pkg
    name pools: ["cell_impl", "sequencer_impl", "cell_impl", "sequencer_impl"]
        
    name pools: ["cell_impl", "sequencer_impl", "cell_impl", "sequencer_impl", "dummy_impl"]
        
    
            rs.name : vec_add_impl
            `define vec_add_impl _9tjr2rhr8kq
            `define vec_add_impl_pkg _9tjr2rhr8kq_pkg
    name pools: ["cell_impl", "sequencer_impl", "cell_impl", "sequencer_impl", "dummy_impl"]
        
    name pools: ["cell_impl", "sequencer_impl", "cell_impl", "sequencer_impl", "dummy_impl", "vec_add_impl"]
        
    
    




package fabric_pkg;
    parameter COLS = 2;
    parameter INSTR_ADDR_WIDTH = 6;
    parameter INSTR_DATA_WIDTH = 32;
    parameter INSTR_HOPS_WIDTH = 4;
    parameter IO_ADDR_WIDTH = 16;
    parameter IO_DATA_WIDTH = 256;
    parameter RESOURCE_INSTR_WIDTH = 27;
    parameter ROWS = 1;
endpackage

module fabric
import fabric_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    input  logic [ROWS-1:0] call,
    output logic [ROWS-1:0] ret,
    output logic [COLS-1:0] io_en_in,
    output logic [COLS-1:0][IO_ADDR_WIDTH-1:0] io_addr_in,
    input  logic [COLS-1:0][IO_DATA_WIDTH-1:0] io_data_in,
    output logic [COLS-1:0] io_en_out,
    output logic [COLS-1:0][IO_ADDR_WIDTH-1:0] io_addr_out,
    output logic [COLS-1:0][IO_DATA_WIDTH-1:0] io_data_out,
    input  logic [ROWS-1:0][INSTR_DATA_WIDTH-1:0] instr_data_in,
    input  logic [ROWS-1:0][INSTR_ADDR_WIDTH-1:0] instr_addr_in,
    input  logic [ROWS-1:0][INSTR_HOPS_WIDTH-1:0] instr_hops_in,
    input  logic [ROWS-1:0] instr_en_in,
    output logic [ROWS-1:0][INSTR_DATA_WIDTH-1:0] instr_data_out,
    output logic [ROWS-1:0][INSTR_ADDR_WIDTH-1:0] instr_addr_out,
    output logic [ROWS-1:0][INSTR_HOPS_WIDTH-1:0] instr_hops_out,
    output logic [ROWS-1:0] instr_en_out
);

    logic [ROWS-1:0][COLS:0] call_net;
    logic [ROWS-1:0][COLS:0] ret_net;
    logic [ROWS-1:0][COLS:0][INSTR_DATA_WIDTH-1:0] instr_data_net;
    logic [ROWS-1:0][COLS:0][INSTR_ADDR_WIDTH-1:0] instr_addr_net;
    logic [ROWS-1:0][COLS:0][INSTR_HOPS_WIDTH-1:0] instr_hops_net;
    logic [ROWS-1:0][COLS:0] instr_en_net;

    always_comb begin
        if (~rst_n) begin
            for (int r=0; r<ROWS; r++) begin
                for (int c=0; c<=COLS; c++) begin
                    call_net[r][c] = 0;
                    ret_net[r][c] = 0;
                    instr_data_net[r][c] = 0;
                    instr_addr_net[r][c] = 0;
                    instr_hops_net[r][c] = 0;
                    instr_en_net[r][c] = 0;
                end
            end
        end
        for (int i=0; i<ROWS; i++) begin
            call_net[i][0] = call[i];
            // ret[i] = ret_net[i][0];
            ret_net[i][COLS] = 1;
            instr_data_net[i][0] = instr_data_in[i];
            instr_addr_net[i][0] = instr_addr_in[i];
            instr_hops_net[i][0] = instr_hops_in[i];
            instr_en_net[i][0] = instr_en_in[i];
            instr_data_out[i] = instr_data_net[i][COLS];
            instr_addr_out[i] = instr_addr_net[i][COLS];
            instr_hops_out[i] = instr_hops_net[i][COLS];
            instr_en_out[i] = instr_en_net[i][COLS];
        end
    end

    for(genvar i=0; i<ROWS; i++) begin
        assign ret[i] = ret_net[i][0];
    end

    cell_impl cell_0_0_inst (
        .clk(clk),
        .rst_n(rst_n),
        .call_in(call_net[0][0]),
        .call_out(call_net[0][1]),
        .ret_in(ret_net[0][1]),
        .ret_out(ret_net[0][0]),
        .io_en_in(io_en_in[0]),
        .io_addr_in(io_addr_in[0]),
        .io_data_in(io_data_in[0]),
        .io_en_out(io_en_out[0]),
        .io_addr_out(io_addr_out[0]),
        .io_data_out(io_data_out[0]),
        .instr_data_in(instr_data_net[0][0]),
        .instr_addr_in(instr_addr_net[0][0]),
        .instr_hops_in(instr_hops_net[0][0]),
        .instr_en_in(instr_en_net[0][0]),
        .instr_data_out(instr_data_net[0][1]),
        .instr_addr_out(instr_addr_net[0][1]),
        .instr_hops_out(instr_hops_net[0][1]),
        .instr_en_out(instr_en_net[0][1])
    );
    cell_impl cell_0_1_inst (
        .clk(clk),
        .rst_n(rst_n),
        .call_in(call_net[0][1]),
        .call_out(call_net[0][2]),
        .ret_in(ret_net[0][2]),
        .ret_out(ret_net[0][1]),
        .io_en_in(io_en_in[1]),
        .io_addr_in(io_addr_in[1]),
        .io_data_in(io_data_in[1]),
        .io_en_out(io_en_out[1]),
        .io_addr_out(io_addr_out[1]),
        .io_data_out(io_data_out[1]),
        .instr_data_in(instr_data_net[0][1]),
        .instr_addr_in(instr_addr_net[0][1]),
        .instr_hops_in(instr_hops_net[0][1]),
        .instr_en_in(instr_en_net[0][1]),
        .instr_data_out(instr_data_net[0][2]),
        .instr_addr_out(instr_addr_net[0][2]),
        .instr_hops_out(instr_hops_net[0][2]),
        .instr_en_out(instr_en_net[0][2])
    );
    
endmodule