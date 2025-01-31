epoch <conv_2_1> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=0, level=0, iter=0, step=0, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=2, level=0, iter=0, step=0, delay=0)

act (mode=0, param=1, ports=0b0101)
wait (cycle=63)

dsu (slot=2, port=3, init_addr=63)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=0)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=4, step=8, delay=2)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)

wait (cycle=5233)

dsu (slot=2, port=3, init_addr=40)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)

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
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=0)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=4, step=0, delay=2)
repx (slot=1, port=2, level=2, iter=-1, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=4)
rep (slot=1, port=1, level=3, iter=4, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=4)
rep (slot=2, port=1, level=3, iter=4, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
rep (slot=4, level=1, iter=3, step=0, delay=18)
rep (slot=4, level=2, iter=4, step=0, delay=20)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=4, step=0, delay=20)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=4, step=0, delay=10)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=16)

wait (cycle=38)

act (ports=4, param=2)

wait (cycle=5)

act (ports=4, param=1)

act (ports=34, param=1)
act (ports=1, param=4)
wait (cycle=13)
act (ports=1, param=3)

wait (cycle=1013)

act (ports=8, param=3)

wait (cycle=4189)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=24, step=2, delay=1)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=24, step=0, delay=1)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=24, step=0, delay=15)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=24, step=1, delay=15)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=1, delay=0)

act (ports=4, param=1)

act (ports=34, param=1)
act (ports=1, param=4)
wait (cycle=13)
act (ports=1, param=3)

wait (cycle=385)

act (ports=8, param=3)


}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=4, step=4, delay=10)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=16)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=21, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=21, step=1, delay=0)

wait (cycle=1105)

act (mode=0, param=2, ports=4)

wait (cycle=4605)

dsu (slot=2, port=2, init_addr=20)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=4)

wait (cycle=3)

act (mode=0, param=1, ports=10)


}
}

}