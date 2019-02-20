% Eq.[4]
% Original       : C = F*C + I*G
% Implementation : C = F*C + I*G

COMPUTING_DRRA_ROW = 2;
COMPUTING_DRRA_COL = 4;
HIDDEN_SIZE = 320;
REG_DEPTH = 32;

C_sram_0 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,0]
F_sram_0 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,0]
I_sram_0 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,0]
G_sram_0 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,0]
C_sram_1 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,1]
F_sram_1 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,1]
I_sram_1 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,1]
G_sram_1 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,1]
C_sram_2 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,2]
F_sram_2 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,2]
I_sram_2 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,2]
G_sram_2 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,2]
C_sram_3 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,3]
F_sram_3 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,3]
I_sram_3 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,3]
G_sram_3 = ones(1, HIDDEN_SIZE/COMPUTING_DRRA_COL);     %! MEM [0,3]

C_reg_0 = zeros(1, REG_DEPTH/2);  %! RFILE [0,0]
C_reg_1 = zeros(1, REG_DEPTH/2);  %! RFILE [0,1]
C_reg_2 = zeros(1, REG_DEPTH/2);  %! RFILE [0,2]
C_reg_3 = zeros(1, REG_DEPTH/2);  %! RFILE [0,3]
F_reg_0 = zeros(1, REG_DEPTH/2);  %! RFILE [0,0]
F_reg_1 = zeros(1, REG_DEPTH/2);  %! RFILE [0,1]
F_reg_2 = zeros(1, REG_DEPTH/2);  %! RFILE [0,2]
F_reg_3 = zeros(1, REG_DEPTH/2);  %! RFILE [0,3]
I_reg_0 = zeros(1, REG_DEPTH/2);  %! RFILE [1,0]
I_reg_1 = zeros(1, REG_DEPTH/2);  %! RFILE [1,1]
I_reg_2 = zeros(1, REG_DEPTH/2);  %! RFILE [1,2]
I_reg_3 = zeros(1, REG_DEPTH/2);  %! RFILE [1,3]
G_reg_0 = zeros(1, REG_DEPTH/2);  %! RFILE [1,0]
G_reg_1 = zeros(1, REG_DEPTH/2);  %! RFILE [1,1]
G_reg_2 = zeros(1, REG_DEPTH/2);  %! RFILE [1,2]
G_reg_3 = zeros(1, REG_DEPTH/2);  %! RFILE [1,3]

for i = 1 : HIDDEN_SIZE/(REG_DEPTH/2)/COMPUTING_DRRA_COL
	C_reg_0 = C_sram_0(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	F_reg_0 = F_sram_0(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	I_reg_0 = I_sram_0(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	G_reg_0 = G_sram_0(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	C_reg_0 = F_reg_0 .* C_reg_0; %! DPU [0,0]
	I_reg_0 = I_reg_0 .* G_reg_0; %! DPU [1,0]
	C_reg_0 = C_reg_0 + I_reg_0;  %! DPU [0,0]

	C_reg_1 = C_sram_1(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	F_reg_1 = F_sram_1(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	I_reg_1 = I_sram_1(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	G_reg_1 = G_sram_1(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	C_reg_1 = F_reg_1 .* C_reg_1; %! DPU [0,1]
	I_reg_1 = I_reg_1 .* G_reg_1; %! DPU [1,1]
	C_reg_1 = C_reg_1 + I_reg_1;  %! DPU [0,1]

	C_reg_2 = C_sram_2(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	F_reg_2 = F_sram_2(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	I_reg_2 = I_sram_2(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	G_reg_2 = G_sram_2(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	C_reg_2 = F_reg_2 .* C_reg_2; %! DPU [0,2]
	I_reg_2 = I_reg_2 .* G_reg_2; %! DPU [1,2]
	C_reg_2 = C_reg_2 + I_reg_2;  %! DPU [0,2]

	C_reg_3 = C_sram_3(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	F_reg_3 = F_sram_3(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	I_reg_3 = I_sram_3(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	G_reg_3 = G_sram_3(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2);
	C_reg_3 = F_reg_3 .* C_reg_3; %! DPU [0,3]
	I_reg_3 = I_reg_3 .* G_reg_3; %! DPU [1,3]
	C_reg_3 = C_reg_3 + I_reg_3;  %! DPU [0,3]

	C_sram_0(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2) = C_reg_0;
	C_sram_1(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2) = C_reg_1;
	C_sram_2(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2) = C_reg_2;
	C_sram_3(i*REG_DEPTH/2-REG_DEPTH/2+1:i*REG_DEPTH/2) = C_reg_3;
end