
module fabric_tb;
import fabric_pkg::*;

    // stimuli
    logic clk;
    logic rst_n;
    logic [ROWS-1:0] call;
    logic [ROWS-1:0] ret;
    logic [COLS-1:0] io_en_in;
    logic [COLS-1:0][IO_ADDR_WIDTH-1:0] io_addr_in;
    logic [COLS-1:0][IO_DATA_WIDTH-1:0] io_data_in;
    logic [COLS-1:0] io_en_out;
    logic [COLS-1:0][IO_ADDR_WIDTH-1:0] io_addr_out;
    logic [COLS-1:0][IO_DATA_WIDTH-1:0] io_data_out;
    
    logic [ROWS-1:0][INSTR_DATA_WIDTH-1:0] instr_data_in;
    logic [ROWS-1:0][INSTR_ADDR_WIDTH-1:0] instr_addr_in;
    logic [ROWS-1:0][INSTR_HOPS_WIDTH-1:0] instr_hops_in;
    logic [ROWS-1:0] instr_en_in;
    logic [ROWS-1:0][INSTR_DATA_WIDTH-1:0] instr_data_out;
    logic [ROWS-1:0][INSTR_ADDR_WIDTH-1:0] instr_addr_out;
    logic [ROWS-1:0][INSTR_HOPS_WIDTH-1:0] instr_hops_out;
    logic [ROWS-1:0] instr_en_out;

    logic [255:0] input_buffer [int];
    logic [255:0] output_buffer [int];

    logic ret_all;
    assign ret_all = &ret;

    int fd;
    int r, c;
    int index;
    string line;
    logic [INSTR_DATA_WIDTH-1:0] temp_instruction;
    realtime start_time, end_time;
    logic [15:0][15:0] ob_line, ib_line;
    initial begin
        rst_n = 0;
        for (int i=0; i<ROWS; i++) begin
            call[i] = 0;
        end
        for (int i=0; i<COLS; i++) begin
            io_data_in[i] = 0;
        end

        @(negedge clk) rst_n = 1;

        // load instructions
        index=0;
        r=0;
        c=0;
        fd = $fopen("instr.bin", "r");
        while (!$feof(fd)) begin
            if ($fscanf(fd, "cell %d %d", r, c)) begin
                $display("cell %d %d", r, c);
                index = 0;
            end else if ($fscanf(fd, "%b", temp_instruction)) begin
                $display("instr_data_in[%d][%d] = %b", r, c, temp_instruction);
                instr_data_in[r] = temp_instruction;
                instr_addr_in[r] = index;
                instr_hops_in[r] = c;
                instr_en_in[r] = 1;
                index = index + 1;
                @ (negedge clk);
                instr_data_in[r] = 0;
                instr_addr_in[r] = 0;
                instr_hops_in[r] = 0;
                instr_en_in[r] = 0;
            end
        end
        
        // record simulation time
        @(negedge clk)
        for (int i=0; i<ROWS; i++) begin
            call[i] = 1;
        end
        start_time = $realtime;
        @(negedge clk)
        for (int i=0; i<ROWS; i++) begin
            call[i] = 0;
        end
        // wait until every cell is called and ret signal is stable
        for(int i=0; i<COLS*2; i++) begin
            @(negedge clk);
        end
        
        // wait for ret to be 1
        @(posedge ret_all);
        // record simulation time
        @(negedge clk) end_time = $realtime;
        $display("Simulation ends! Total cycles = %d", (end_time - start_time)/10);

        // display all the output buffer and write it to a file
        $display("Output Buffers:");
        fd = $fopen("sram_image_out.bin", "w+");
        foreach(output_buffer[i]) begin
            for (int x = 0; x < 16; x = x + 1) begin
                ob_line[x] = output_buffer[i][16*x +: 16];
            end
            $display("OB[%d] = %s", i, $sformatf("%p", ob_line));
            $fwrite(fd, "%d %b\n", i, output_buffer[i]);
        end
        $finish;
    end

    // loading input buffer
    logic [255:0] temp_data;
    initial begin
        fd = $fopen("sram_image_in.bin", "r");
        // for each line put it in the input buffer
        $display("Loading input buffer");
        while (!$feof(fd)) begin
            $fscanf(fd, "%d %b", index, temp_data);
            $display("index = %d, data = %b", index, temp_data);
            input_buffer[index] = temp_data;
        end

        $display("Input Buffers:");
        foreach(input_buffer[i]) begin
            for (int x = 0; x < 16; x = x + 1) begin
                ib_line[x] = input_buffer[i][16*x +: 16];
            end
            $display("IB[%d] = %s", i, $sformatf("%p", ib_line));
        end
    end

    // DUT
    fabric fabric_inst (
        .clk(clk),
        .rst_n(rst_n),
        .call(call),
        .ret(ret),
        .io_en_in(io_en_in),
        .io_addr_in(io_addr_in),
        .io_data_in(io_data_in),
        .io_en_out(io_en_out),
        .io_addr_out(io_addr_out),
        .io_data_out(io_data_out),
        .instr_data_in(instr_data_in),
        .instr_addr_in(instr_addr_in),
        .instr_hops_in(instr_hops_in),
        .instr_en_in(instr_en_in),
        .instr_data_out(instr_data_out),
        .instr_addr_out(instr_addr_out),
        .instr_hops_out(instr_hops_out),
        .instr_en_out(instr_en_out)
    );

    // clock
    initial begin
        clk = 0;
        forever begin
        #5 clk = ~clk;
        end
    end

    // memory
    always @(posedge clk) begin
        for (int i=0; i<COLS; i++) begin
            if (io_en_out[i]) begin
                output_buffer[io_addr_out[i]] = io_data_out[i];
            end
            if (io_en_in[i]) begin
                io_data_in[i] = input_buffer[io_addr_in[i]];
            end
        end
    end

endmodule