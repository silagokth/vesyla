epoch <rb0> {

cell (x=0, y=0){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, step=0, iter=1, delay=0)
repx (slot=1, port=0, level=0, step=1, iter=-1, delay=0)
rep (slot=1, port=0, level=1, step=1, iter=15, delay=0)
rep (slot=1, port=0, level=2, step=16, iter=0, delay=0)
repx (slot=1, port=0, level=2, step=0, iter=-1, delay=7)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=31, delay=0)
repx (slot=1, port=2, level=0, step=0, iter=-1, delay=0)
rep (slot=1, port=2, level=1, step=0, iter=0, delay=0)
repx (slot=1, port=2, level=1, step=0, iter=-1, delay=7)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=7, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=8, delay=56)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=0)
rep (slot=2, port=3, level=2, iter=0, step=0, delay=56)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)

wait (cycle=6)

act (mode=0, param=1, ports=0b0101)
act (mode=0, param=2, ports=0b1000)

halt
}
}
cell (x=0, y=1){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=16)
rep (slot=1, port=0, level=0, step=0, iter=1, delay=0)
repx (slot=1, port=0, level=0, step=1, iter=-1, delay=0)
rep (slot=1, port=0, level=1, step=1, iter=15, delay=0)
rep (slot=1, port=0, level=2, step=16, iter=0, delay=0)
repx (slot=1, port=0, level=2, step=0, iter=-1, delay=7)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=31, delay=0)
repx (slot=1, port=2, level=0, step=0, iter=-1, delay=0)
rep (slot=1, port=2, level=1, step=0, iter=0, delay=0)
repx (slot=1, port=2, level=1, step=0, iter=-1, delay=7)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=7, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=8, delay=56)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=0)
rep (slot=2, port=3, level=2, iter=0, step=0, delay=56)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)

wait (cycle=6)

act (mode=0, param=1, ports=0b0101)
act (mode=0, param=2, ports=0b1000)

halt
}
}
cell (x=0, y=2){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=32)
rep (slot=1, port=0, level=0, step=0, iter=1, delay=0)
repx (slot=1, port=0, level=0, step=1, iter=-1, delay=0)
rep (slot=1, port=0, level=1, step=1, iter=15, delay=0)
rep (slot=1, port=0, level=2, step=16, iter=0, delay=0)
repx (slot=1, port=0, level=2, step=0, iter=-1, delay=7)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=31, delay=0)
repx (slot=1, port=2, level=0, step=0, iter=-1, delay=0)
rep (slot=1, port=2, level=1, step=0, iter=0, delay=0)
repx (slot=1, port=2, level=1, step=0, iter=-1, delay=7)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=7, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=8, delay=56)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=0)
rep (slot=2, port=3, level=2, iter=0, step=0, delay=56)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)

wait (cycle=6)

act (mode=0, param=1, ports=0b0101)
act (mode=0, param=2, ports=0b1000)

halt
}
}
cell (x=0, y=3){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=48)
rep (slot=1, port=0, level=0, step=0, iter=1, delay=0)
repx (slot=1, port=0, level=0, step=1, iter=-1, delay=0)
rep (slot=1, port=0, level=1, step=1, iter=15, delay=0)
rep (slot=1, port=0, level=2, step=16, iter=0, delay=0)
repx (slot=1, port=0, level=2, step=0, iter=-1, delay=7)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=31, delay=0)
repx (slot=1, port=2, level=0, step=0, iter=-1, delay=0)
rep (slot=1, port=2, level=1, step=0, iter=0, delay=0)
repx (slot=1, port=2, level=1, step=0, iter=-1, delay=7)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=7, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=8, delay=56)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=0)
rep (slot=2, port=3, level=2, iter=0, step=0, delay=56)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)

wait (cycle=6)

act (mode=0, param=1, ports=0b0101)
act (mode=0, param=2, ports=0b1000)




halt
}
}
cell (x=1, y=0){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=0, sr=0, source=3, target= 128)
act (mode=0, param=0, ports=0b0100)

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)
act (ports=1)


dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)

dpu (slot=4, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=3, step=0, delay=0)

act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=58)


dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=3, step=0, delay=60)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=0)

act (ports=8, param=3)

halt
}
}
cell (x=1, y=1){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=0, sr=0, source=3, target= 128)
act (mode=0, param=0, ports=0b0100)

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)
act (ports=1)


dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)

dpu (slot=4, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=3, step=0, delay=0)

act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=58)


dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=3, step=0, delay=60)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=0)

act (ports=8, param=3)

halt
}
}
cell (x=1, y=2){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=0, sr=0, source=3, target= 128)
act (mode=0, param=0, ports=0b0100)

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)
act (ports=1)


dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)

dpu (slot=4, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=3, step=0, delay=0)

act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=58)


dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=3, step=0, delay=60)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=0)

act (ports=8, param=3)

halt
}
}
cell (x=1, y=3){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=0, sr=0, source=3, target= 128)
act (mode=0, param=0, ports=0b0100)

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)
act (ports=1)


dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)

dpu (slot=4, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=3, step=0, delay=0)

act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=58)


dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=3, step=0, delay=60)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=0)

act (ports=8, param=3)


halt
}
}
cell (x=2, y=0){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b0100)
act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=3, step=4, delay=60)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=3, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=3, level=1, iter=-1, step=0, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=1, level=1, iter=-1, step=0, delay=0)

wait (cycle=82)

act (ports=0b0100, param=2)
act (ports=0b1010, param=1)

halt
}
}
cell (x=2, y=1){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b0100)
act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=3, step=4, delay=60)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=3, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=3, level=1, iter=-1, step=0, delay=0)
dsu (slot=1, port=1, init_addr=16)
rep (slot=1, port=1, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=1, level=1, iter=-1, step=0, delay=0)

wait (cycle=82)

act (ports=0b0100, param=2)
act (ports=0b1010, param=1)

halt
}
}
cell (x=2, y=2){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b0100)
act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=3, step=4, delay=60)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=3, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=3, level=1, iter=-1, step=0, delay=0)
dsu (slot=1, port=1, init_addr=32)
rep (slot=1, port=1, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=1, level=1, iter=-1, step=0, delay=0)

wait (cycle=82)

act (ports=0b0100, param=2)
act (ports=0b1010, param=1)

halt
}
}
cell (x=2, y=3){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b0100)
act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=3, step=4, delay=60)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=3, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=3, level=1, iter=-1, step=0, delay=0)
dsu (slot=1, port=1, init_addr=48)
rep (slot=1, port=1, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=3, step=4, delay=60)
repx (slot=1, port=1, level=1, iter=-1, step=0, delay=0)

wait (cycle=82)

act (ports=0b0100, param=2)
act (ports=0b1010, param=1)
    halt
}
}
}