epoch <out_dense> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=20, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=20, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=15)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#26

wait (cycle=0)

dsu (slot=2, port=3, init_addr=4)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=4, delay=60)

act (mode=0, param=2, ports=0b1000)
#31

}

}

cell (x=1, y=0) {

raw {

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target= 6)
route (slot=0, option=0, sr=0, source=3, target= 128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=4, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=1, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=4, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=1, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=4, step=0, delay=0)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=4, step=0, delay=63)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=4, step=1, delay=63)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=1, delay=0)

act (ports=4, param=1)
#27

wait (cycle=3)

act (ports=4, param=2)
#32

act (ports=34, param=1)
act (ports=1, param=4)

wait (cycle=62)

act (ports=1, param=3)
#98

wait (cycle=191)

act (ports=8, param=3)
#291


}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=1, step=1, delay=0)

wait (cycle=282)

act (mode=0, param=2, ports=4)
#292

act (mode=0, param=1, ports=10)


}
}

}