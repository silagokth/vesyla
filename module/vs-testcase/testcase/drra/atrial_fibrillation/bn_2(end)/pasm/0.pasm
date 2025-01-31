epoch <bn_2> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=23, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=23, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

wait (cycle=45)

dsu (slot=2, port=3, init_addr=22)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)

dsu (slot=2, port=3, init_addr=23)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)

wait (cycle=361)

dsu (slot=2, port=3, init_addr=20)
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
}

}

cell (x=1, y=0) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=4, target=8)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=7, target=9)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target= 22)
route (slot=0, option=0, sr=0, source=9, target= 128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=4, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=4, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=4, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=4, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=4, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=4, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=4, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=4, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

wait (cycle=7)

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)

wait (cycle=3)

act (ports=4, param=1)

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)

wait (cycle=59)

act (ports=8, param=9)

wait (cycle=278)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=24, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=24, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=24, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=24, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=24, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=24, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=1, step=1, delay=0)

act (ports=4, param=1)

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)

wait (cycle=22)

act (ports=8, param=9)

}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=4, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=21, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=21, step=1, delay=0)

wait (cycle=121)

act (mode=0, param=2, ports=4)

wait (cycle=324)

dsu (slot=2, port=2, init_addr=20)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=4)

wait (cycle=2)

act (mode=0, param=1, ports=10)

}
}

}