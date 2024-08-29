epoch <relu_3> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
#8

wait (cycle=6)

act (mode=0, param=1, ports=0b0101)

act (mode=0, param=2, ports=0b1000)
#17




}

}

cell (x=1, y=0) {

raw {

swb (slot=0, option=0, source=1, target=3)
swb (slot=0, option=0, source=3, target=2)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target= 2)
route (slot=0, option=0, sr=0, source=2, target= 128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=1, step=0, delay=0)

dpu (slot=3, option=0, mode=23)
dsu (slot=2, port=0, init_addr=0)
rep (slot=2, port=0, level=0, iter=0, step=1, delay=0)
repx (slot=2, port=0, level=0, iter=1, step=0, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)

act (ports=4, param=1)
#18

act (ports=2, param=1)
act (ports=1, param=3)

act (ports=1, param=2)
#21

wait (cycle=59)

act (ports=8, param=2)
#81





}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
#8

wait (cycle=73)

act (mode=0, param=2, ports=4)
#82

act (mode=0, param=1, ports=10)





}
}

}