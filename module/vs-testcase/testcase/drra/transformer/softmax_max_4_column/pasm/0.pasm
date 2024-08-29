epoch <rb0> {

cell (x=0, y=0){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, step=1, iter=4, delay=0)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=4, delay=0)


dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=8, delay=56)

act (mode=0, param=1, ports=0b0101)
wait (cycle = 11)

act (mode=0, param=2, ports=0b1000)

halt
}
}
cell (x=0, y=1){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=4)
rep (slot=1, port=0, level=0, step=1, iter=4, delay=0)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=4, delay=0)


dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=8, delay=56)

act (mode=0, param=1, ports=0b0101)
wait (cycle = 8)

act (mode=0, param=2, ports=0b1000)


halt
}
}
cell (x=0, y=2){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=8)
rep (slot=1, port=0, level=0, step=1, iter=4, delay=0)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=4, delay=0)


dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=8, delay=56)

act (mode=0, param=1, ports=0b0101)
wait (cycle = 11)

act (mode=0, param=2, ports=0b1000)

halt
}
}
cell (x=0, y=3){
raw{
route (slot=0, option=0, sr=0, source=2, target= 128)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=12)
rep (slot=1, port=0, level=0, step=1, iter=4, delay=0)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, step=1, iter=4, delay=0)


dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=8, delay=56)

act (mode=0, param=1, ports=0b0101)
wait (cycle = 8)

act (mode=0, param=2, ports=0b1000)

halt
}
}
cell (x=1, y=0){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=1, sr=0, source=3, target= 0b100000000)
route (slot=0, option=1, sr=1, source=5, target= 0b00110)
fsm (slot=0, port=2, delay_0=20)
rep (slot=0, port=2, level=0, iter=1, delay=10)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=1)


swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (ports=1)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)

dpu (slot=4, option=0, mode=13)

dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=1, step=1, delay=0)

act (mode=0, param=0, ports=0b0100)
act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
wait (cycle = 28)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=1)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=1, step=1, delay=0)
act (ports=4, param=1)
act (ports=2, param=1)
wait (cycle = 0)
act (ports=1, param=3)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=0, delay=0)

act (ports=0b1000, param=3)

halt
}
}
cell (x=1, y=1){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=0, sr=0, source=3, target= 0b001000)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=1)

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (ports=1)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)

dpu (slot=4, option=0, mode=13)

dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=1, step=1, delay=0)

act (mode=0, param=0, ports=0b0100)
act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
wait (cycle = 30)
act (ports=1, param=3)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=0, delay=0)

act (ports=0b1000, param=3)




halt
}
}
cell (x=1, y=2){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=1, sr=0, source=3, target= 0b1000000)
route (slot=0, option=1, sr=1, source=5, target= 0b00110)
fsm (slot=0, port=2, delay_0=20)
rep (slot=0, port=2, level=0, iter=1, delay=10)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=1)


swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (ports=1)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)

dpu (slot=4, option=0, mode=13)

dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=1, step=1, delay=0)

act (mode=0, param=0, ports=0b0100)
act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
wait (cycle = 28)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=1)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=1, step=1, delay=0)
act (ports=4, param=1)
act (ports=2, param=1)
wait (cycle = 0)
act (ports=1, param=3)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=0, delay=0)

act (ports=0b1000, param=3)

halt
}
}
cell (x=1, y=3){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b00110)
route (slot=0, option=0, sr=0, source=3, target= 0b001000)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=1)

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (ports=1)


dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=32, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)

dpu (slot=4, option=0, mode=13)

dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=1, step=1, delay=0)

act (mode=0, param=0, ports=0b0100)
act (mode=0, ports=4, param=1)
act (mode=0, ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=4)
wait (cycle = 30)
act (ports=1, param=3)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=0, delay=0)

act (ports=0b1000, param=3)

halt
}
}
cell (x=2, y=0){
raw{
halt
}
}
cell (x=2, y=1){
raw{
route (slot=0, option=0, sr=1, source=0, target= 0b00110)
route (slot=0, option=1, sr=1, source=2, target= 0b00110)
route (slot=0, option=1, sr=0, source=3, target= 128)
fsm (slot=0, port=2, delay_0=1)
rep (slot=0, port=2, level=0, iter=1, delay=0)

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=4, target=3)
act (ports=1)


dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=2, step=16, delay=1)

dpu (slot=4, option=0, mode=13)

dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=1, step=1, delay=0)

wait (cycle = 49)

act (mode=0, param=0, ports=0b0100)
act (mode=0, ports=4, param=1)

act (ports=2, param=1)
act (ports=1, param=4)
wait (cycle = 2)
act (ports=1, param=3)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=0, delay=0)

act (ports=0b1000, param=3)

halt
}
}
cell (x=2, y=2){
raw{
halt
}
}
cell (x=2, y=3){
raw{
halt
}
}
cell (x=3, y=0){
raw{
halt
}
}
cell (x=3, y=1){
raw{
route (slot=0, option=0, sr=1, source=1, target= 0b0000000000000100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=0, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=1, step=1, delay=0)

act (mode=0, param=0, ports=0b0100)
wait (cycle = 67)

act (ports=0b0100, param=2)
act (ports=0b1010, param=1)

halt
}
}
cell (x=3, y=2){
raw{
halt
}
}
cell (x=3, y=3){
raw{
    halt
}
}
}