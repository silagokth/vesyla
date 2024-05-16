epoch rb0 {
cell 0_0

rop route0 slot=0, port=2
route slot=0, option=0, sr=0, source=2, target= 0b010000000

rop input_r slot=1, port=0
dsu slot=1, port=0, init_addr=0
rep slot=1, port=0, level=0, iter=3, step=1, delay=0

rop input_w slot=1, port=2
dsu slot=1, port=2, init_addr=0
rep slot=1, port=2, level=0, iter=3, step=1, delay=0

rop io_r2 slot=2, port=3
dsu slot=2, port=3, init_addr=0
rep slot=2, port=3, level=0, iter=1, step=0, delay=0

rop io_r1 slot=2, port=3
dsu slot=2, port=3, init_addr=1
rep slot=2, port=3, level=0, iter=2, step=1, delay=0

cell 1_0
rop swb1 slot=0, port=0
swb slot=0, option=0, source=1, target=4
swb slot=0, option=0, source=2, target=5
swb slot=0, option=0, source=4, target=3

rop route1 slot=0, port=2
route slot=0, option=0, sr=1, source=1, target= 6
route slot=0, option=0, sr=0, source=3, target= 128

rop write_1w slot=1, port=2
dsu slot=1, port=2, init_addr=0
rep slot=1, port=2, level=0, iter=2, step=1, delay=0

rop write_2w slot=2, port=2
dsu slot=2, port=2, init_addr=0
rep slot=2, port=2, level=0, iter=1, step=1, delay=0

rop read_1n slot=1, port=1
dsu slot=1, port=1, init_addr=0
rep slot=1, port=1, level=0, iter=3, step=1, delay=0
rep slot=1, port=1, level=1, iter=30, step=1, delay=t1

rop read_2n slot=2, port=1
dsu slot=2, port=1, init_addr=0
rep slot=2, port=1, level=0, iter=3, step=1, delay=0
rep slot=2, port=1, level=1, iter=30, step=0, delay=t1

rop compute slot=4, port=0
dpu slot=4, option=0, mode=2
rep slot=4, level=0, iter=30, step=0, delay=t2

rop write_3n slot=3, port=0
dsu slot=3, port=0, init_addr=1
rep slot=3, port=0, level=0, iter=30, step=1, delay=t3

rop read_3w slot=3, port=3
dsu slot=3, port=3, init_addr=0
rep slot=3, port=3, level=0, iter=2, step=1, delay=0


cell 2_0

rop route2 slot=0, port=2
route slot=0, option=0, sr=1, source=1, target= 4

rop io_w slot=2, port=2
dsu slot=2, port=2, init_addr=0
rep slot=2, port=2, level=0, iter=2, step=1, delay=0

rop output_r slot=1, port=3
dsu slot=1, port=3, init_addr=0
rep slot=1, port=3, level=0, iter=2, step=1, delay=0

rop output_w slot=1, port=1
dsu slot=1, port=1, init_addr=0
rep slot=1, port=1, level=0, iter=2, step=1, delay=0
}
