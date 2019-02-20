
% Calculation with accumulation testcase

% Features:
%  - Using different accumulation operations on the same cell
%  - Utilizing fixed-point values

A      = fi([0.012024 0.012024 0.012146 0.012421 0.012665 0.012817 0.011780 0.011383 0.011627 0.012817 0.013733 0.014130 0.014374 0.014893 0.016998 0.012146 0.006653 0.004700 0.006134 0.007721 0.007050 0.005493 0.006134 0.005890 0.005371 0.004974 0.004700 0.004700 0.004822 0.004974 0.006134 0.005890], 1, 16, 15, 'RoundingMethod', 'Floor'); %! RFILE<> [0,0]
B      = fi([0.006012 0.006409 0.006409 0.006134 0.005615 0.005096 0.004456 0.004181 0.004456 0.005096 0.005096 0.004456 0.004181 0.004456 0.005737 0.005737 0.005737 0.005737 0.005737 0.005737 0.005737 0.005737 0.004822 0.004822 0.004974 0.005219 0.005371 0.005615 0.005737 0.005890 0.005493 0.005493], 1, 16, 15, 'RoundingMethod', 'Floor'); %! RFILE<> [0,1]
C 	   = fi([0.001831 0.001831 0.001923 0.001953 0.002045 0.002075 0.002167 0.002167 0.002197 0.002289 0.002441 0.002319 0.010010 0.009918 0.009949 0.010010 0.002991 0.002716 0.002167 0.002289 0.002228 0.002228 0.002014 0.001953 0.001923 0.001892 0.001801 0.001740 0.001678 0.001678 0.001740 0.001740], 1, 16, 15, 'RoundingMethod', 'Floor'); %! RFILE<> [0,2]
result = fi([0 0 0], 1, 16, 15, 'RoundingMethod', 'Floor'); %! RFILE<> [1,1]

result(1) = sum(abs(A - B)); %! DPU [1,1]
result(2) = sum(fi(A .* B, 1, 16, 15, 'RoundingMethod', 'Floor'));   	 %! DPU [1,1]
result(3) = sum(fi(A .* (B + C), 1, 16, 15, 'RoundingMethod', 'Floor')); %! DPU [1,1]