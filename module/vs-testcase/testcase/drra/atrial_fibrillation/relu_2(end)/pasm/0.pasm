epoch <relu_2> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=5, step=4, delay=60)
#9

wait (cycle=9)

act (mode=0, param=1, ports=0b0101)

act (mode=0, param=2, ports=0b1000)
#21

wait (cycle=316)

dsu (slot=2, port=3, init_addr=20)
#339
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#341



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
rep (slot=1, port=2, level=1, iter=5, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=1, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=5, step=0, delay=0)

dpu (slot=3, option=0, mode=23)
dsu (slot=2, port=0, init_addr=0)
rep (slot=2, port=0, level=0, iter=0, step=1, delay=0)
repx (slot=2, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=2, port=0, level=1, iter=5, step=0, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=5, step=0, delay=60)

act (ports=4, param=1)
#22

act (ports=2, param=1)
act (ports=1, param=3)

act (ports=1, param=2)
#25

wait (cycle=59)

act (ports=8, param=2)
#85

wait (cycle=245)

dsu (slot=1, port=2, init_addr=0)
#333
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=25, step=1, delay=0)

dpu (slot=3, option=0, mode=23)
dsu (slot=2, port=0, init_addr=0)
rep (slot=2, port=0, level=0, iter=25, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#342

act (ports=2, param=1)
act (ports=1, param=3)

act (ports=1, param=2)
#345

wait (cycle=23)

act (ports=8, param=2)
#369




}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=60)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=22, step=1, delay=0)
#9

wait (cycle=76)

act (mode=0, param=2, ports=4)
#86

wait (cycle=280)

dsu (slot=2, port=2, init_addr=20)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#370

act (mode=0, param=1, ports=10)




}
}

}