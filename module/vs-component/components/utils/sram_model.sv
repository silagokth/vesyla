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