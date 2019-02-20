Wr_reg = [4096, 4076, 4017, 3919, 3784, 3612, 3405, 3166, 2896, 2598, 2275, 1930, 1567, 1189, 799, 401, 0, -401, -799, -1189, -1567, -1930, -2275, -2598, -2896, -3166, -3405, -3612, -3784, -3919, -4017, -4076]; %! RFILE [0,1]
Wi_reg = [0, -401, -799, -1189, -1567, -1930, -2275, -2598, -2896, -3166, -3405, -3612, -3784, -3919, -4017, -4076, -4096, -4076, -4017, -3919, -3784, -3612, -3405, -3166, -2896, -2598, -2275, -1930, -1567, -1189, -799, -401]; %! RFILE [0,1]
Dr0_reg = [1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448, 1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448, 1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448, 1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448, 1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448, 1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448, 1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448, 1448, 1448, 2648, 4345, 1448, -4345, -5544, -1448]; %! RFILE [0,0]
Di0_reg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; %! RFILE [1,0]

N=64;
STAGE=6;
STAGEADD1=STAGE+1;
SHIFTLOG2N = 96;

__sl_dpu_load(Wr_reg, Wi_reg); %! DPU [0,0]

for i=1:STAGE
	a=i+1;  %!RACCU_VAR
	c=a+SHIFTLOG2N; %!RACCU_VAR
	b=STAGEADD1-a;  %!RACCU_VAR
	[__sl_agu_bitreverse(Dr0_reg(1:2:N-1), b), __sl_agu_bitreverse(Di0_reg(1:2:N-1), b), __sl_agu_bitreverse(Dr0_reg(2:2:N), b), __sl_agu_bitreverse(Di0_reg(2:2:N), b)] = __sl_dpu_butterfly(__sl_agu_bitreverse(Dr0_reg(1:2:N-1), b), __sl_agu_bitreverse(Di0_reg(1:2:N-1), b), __sl_agu_bitreverse(Dr0_reg(2:2:N), b), __sl_agu_bitreverse(Di0_reg(2:2:N), b), c); %! DPU<sat> [0,0]
end

__sl_dpu_load(__sl_agu_bitreverse(Dr0_reg, STAGE), __sl_agu_bitreverse(Di0_reg, STAGE)); %! DPU [0,0]
[Dr0_reg, Di0_reg] = __sl_dpu_store(); %! DPU [0,0]