epoch <pool_1_1> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=0, step=1, delay=0)
repx (slot=1, port=0, level=0, iter=1, step=0, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=0, step=1, delay=0)
repx (slot=1, port=2, level=0, iter=1, step=0, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=8, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)
#14

wait (cycle=63)

act (mode=0, param=2, ports=0b1000)
#79

wait (cycle=1028)

dsu (slot=1, port=0, init_addr=64)
#1109
rep (slot=1, port=0, level=0, iter=24, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=24, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=3, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)
#1118

wait (cycle=23)

act (mode=0, param=2, ports=0b1000)
#1143


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
rep (slot=1, port=2, level=2, iter=8, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=8, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=8, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=8, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=8, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=8, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)
#34

wait (cycle=44)

act (ports=68, param=1)
#80

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#83

wait (cycle=122)

act (ports=8, param=3)
#207

wait (cycle=908)

dsu (slot=1, port=2, init_addr=0)
#1117
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=1, port=2, level=2, iter=3, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=3, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=3, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=3, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)

act (ports=68, param=1)
#1144

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#1147

wait (cycle=122)

act (ports=8, param=3)
#1271



}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=8, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=44, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=44, step=1, delay=0)
#10

wait (cycle=196)

act (mode=0, param=2, ports=4)
#208

wait (cycle=1058)

dsu (slot=2, port=2, init_addr=32)
#1267
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=3, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

act (mode=0, param=2, ports=4)
#1272

wait (cycle=269)

act (mode=0, param=1, ports=10)



}
}

}