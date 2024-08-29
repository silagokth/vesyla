epoch <pool_1_2> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=48, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=48, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=6, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)

wait (cycle=47)

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
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=1, port=2, level=2, iter=6, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=6, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=6, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=6, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=6, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=6, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)

wait (cycle=26)

act (ports=68, param=1)

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=122)

act (ports=8, param=3)




}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=6, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=24, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=24, step=1, delay=0)

wait (cycle=178)

act (mode=0, param=2, ports=4)

wait (cycle=667)

act (mode=0, param=1, ports=10)




}
}

cell (x=0, y=1) {

raw {
route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=48)
rep (slot=1, port=0, level=0, iter=40, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=40, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=5, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)

wait (cycle=47)

act (mode=0, param=2, ports=0b1000)


}

}

cell (x=1, y=1) {

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
rep (slot=1, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=1, port=2, level=2, iter=5, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=5, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=5, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=5, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=5, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=5, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)

wait (cycle=26)

act (ports=68, param=1)

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=122)

act (ports=8, param=3)


}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=20, step=1, delay=0)
dsu (slot=1, port=1, init_addr=24)
rep (slot=1, port=1, level=0, iter=20, step=1, delay=0)

wait (cycle=178)

act (mode=0, param=2, ports=4)

wait (cycle=667)

act (mode=0, param=1, ports=10)

    
}

}

}