epoch <dense_2> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=27)

dsu (slot=1, port=0, init_addr=22)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
rep (slot=1, port=0, level=1, iter=32, step=22, delay=15)
repx (slot=1, port=0, level=1, iter=0, step=0, delay=5)
dsu (slot=1, port=2, init_addr=22)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=32, step=0, delay=15)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=5)

dsu (slot=2, port=3, init_addr=0)
#44
rep (slot=2, port=3, level=0, iter=2, step=22, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=2, iter=5, step=4, delay=56)
rep (slot=2, port=3, level=3, iter=32, step=0, delay=29)
repx (slot=2, port=3, level=3, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)

act (mode=0, param=2, ports=0b1000)
#51

wait (cycle=313)

dsu (slot=2, port=3, init_addr=20)
#366
rep (slot=2, port=3, level=0, iter=2, step=22, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=2, iter=32, step=0, delay=33)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=5)

act (mode=0, param=2, ports=0b1000)
#371



}

}

cell (x=1, y=0) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=7, target=4)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target= 6)
route (slot=0, option=0, sr=0, source=4, target= 128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=5, step=0, delay=57)
rep (slot=1, port=2, level=2, iter=32, step=0, delay=30)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=5, step=0, delay=57)
rep (slot=2, port=2, level=2, iter=32, step=0, delay=30)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=1, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=5, step=0, delay=0)
rep (slot=1, port=1, level=2, iter=32, step=0, delay=37)
repx (slot=1, port=1, level=2, iter=0, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=1, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=5, step=0, delay=0)
rep (slot=2, port=1, level=2, iter=32, step=0, delay=37)
repx (slot=2, port=1, level=2, iter=0, step=0, delay=0)

dpu (slot=5, option=0, mode=8)
rep (slot=5, level=0, iter=5, step=0, delay=63)
rep (slot=5, level=1, iter=32, step=0, delay=36)
repx (slot=5, level=1, iter=0, step=0, delay=1)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=5, step=1, delay=63)
rep (slot=3, port=0, level=1, iter=32, step=0, delay=36)
repx (slot=3, port=0, level=1, iter=0, step=0, delay=1)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=6, step=1, delay=0)
rep (slot=3, port=1, level=1, iter=32, step=0, delay=31)
repx (slot=3, port=1, level=1, iter=0, step=0, delay=5)

dpu (slot=7, option=0, mode=10)
rep (slot=7, level=0, iter=32, step=0, delay=36)
repx (slot=7, level=0, iter=0, step=0, delay=5)
dsu (slot=4, port=0, init_addr=0)
rep (slot=4, port=0, level=0, iter=32, step=1, delay=36)
repx (slot=4, port=0, level=0, iter=0, step=0, delay=5)

dsu (slot=4, port=3, init_addr=0)
rep (slot=4, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#52

act (ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=5)
#55
wait (cycle=62)
act (ports=1, param=3)
#119

wait (cycle=228)

dsu (slot=1, port=2, init_addr=0)
#349
rep (slot=1, port=2, level=0, iter=2, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=32, step=0, delay=34)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=5)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=32, step=0, delay=34)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=5)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=24, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=32, step=0, delay=13)
repx (slot=1, port=1, level=1, iter=0, step=0, delay=5)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=24, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=32, step=0, delay=13)
repx (slot=2, port=1, level=1, iter=0, step=0, delay=5)

dpu (slot=5, option=0, mode=8)
rep (slot=5, level=0, iter=32, step=0, delay=36)
repx (slot=5, level=0, iter=0, step=0, delay=5)
dsu (slot=3, port=0, init_addr=5)
rep (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=32, step=0, delay=36)
repx (slot=3, port=0, level=1, iter=0, step=0, delay=5)

act (ports=4, param=1)
#372

act (ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=5)
#375
wait (cycle=22)
act (ports=1, param=3)
#399

act (ports=2, param=3)
act (ports=1, param=7)
#401
wait (cycle=4)
act (ports=1, param=4)
#407

wait (cycle=11069)

act (ports=8, param=4)
#11478




}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=2, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=2, step=1, delay=0)
#8

wait (cycle=11469)

act (mode=0, param=2, ports=4)
#11479

act (mode=0, param=1, ports=10)




}
}

cell (x=0, y=1) {

raw {
route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

wait (cycle=27)

dsu (slot=1, port=0, init_addr=726)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
rep (slot=1, port=0, level=1, iter=32, step=22, delay=15)
repx (slot=1, port=0, level=1, iter=0, step=0, delay=5)
dsu (slot=1, port=2, init_addr=22)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=32, step=0, delay=15)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=5)

dsu (slot=2, port=3, init_addr=0)
#44
rep (slot=2, port=3, level=0, iter=2, step=22, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=2, iter=5, step=4, delay=56)
rep (slot=2, port=3, level=3, iter=32, step=0, delay=29)
repx (slot=2, port=3, level=3, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)

act (mode=0, param=2, ports=0b1000)
#51

wait (cycle=313)

dsu (slot=2, port=3, init_addr=20)
#366
rep (slot=2, port=3, level=0, iter=2, step=22, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=2, iter=32, step=0, delay=33)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=5)

act (mode=0, param=2, ports=0b1000)
#371


}

}

cell (x=1, y=1) {

raw {
swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=7, target=4)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target= 6)
route (slot=0, option=0, sr=0, source=4, target= 128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=5, step=0, delay=57)
rep (slot=1, port=2, level=2, iter=32, step=0, delay=30)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=5, step=0, delay=57)
rep (slot=2, port=2, level=2, iter=32, step=0, delay=30)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=1, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=5, step=0, delay=0)
rep (slot=1, port=1, level=2, iter=32, step=0, delay=37)
repx (slot=1, port=1, level=2, iter=0, step=0, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=0, step=1, delay=0)
repx (slot=2, port=1, level=0, iter=1, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=5, step=0, delay=0)
rep (slot=2, port=1, level=2, iter=32, step=0, delay=37)
repx (slot=2, port=1, level=2, iter=0, step=0, delay=0)

dpu (slot=5, option=0, mode=8)
rep (slot=5, level=0, iter=5, step=0, delay=63)
rep (slot=5, level=1, iter=32, step=0, delay=36)
repx (slot=5, level=1, iter=0, step=0, delay=1)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=5, step=1, delay=63)
rep (slot=3, port=0, level=1, iter=32, step=0, delay=36)
repx (slot=3, port=0, level=1, iter=0, step=0, delay=1)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=6, step=1, delay=0)
rep (slot=3, port=1, level=1, iter=32, step=0, delay=31)
repx (slot=3, port=1, level=1, iter=0, step=0, delay=5)

dpu (slot=7, option=0, mode=10)
rep (slot=7, level=0, iter=32, step=0, delay=36)
repx (slot=7, level=0, iter=0, step=0, delay=5)
dsu (slot=4, port=0, init_addr=0)
rep (slot=4, port=0, level=0, iter=32, step=1, delay=36)
repx (slot=4, port=0, level=0, iter=0, step=0, delay=5)

dsu (slot=4, port=3, init_addr=0)
rep (slot=4, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#52

act (ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=5)
#55
wait (cycle=62)
act (ports=1, param=3)
#119

wait (cycle=228)

dsu (slot=1, port=2, init_addr=0)
#349
rep (slot=1, port=2, level=0, iter=2, step=1, delay=1)
rep (slot=1, port=2, level=1, iter=32, step=0, delay=34)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=5)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=1)
rep (slot=2, port=2, level=1, iter=32, step=0, delay=34)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=5)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=24, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=32, step=0, delay=13)
repx (slot=1, port=1, level=1, iter=0, step=0, delay=5)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=24, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=32, step=0, delay=13)
repx (slot=2, port=1, level=1, iter=0, step=0, delay=5)

dpu (slot=5, option=0, mode=8)
rep (slot=5, level=0, iter=32, step=0, delay=36)
repx (slot=5, level=0, iter=0, step=0, delay=5)
dsu (slot=3, port=0, init_addr=5)
rep (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=32, step=0, delay=36)
repx (slot=3, port=0, level=1, iter=0, step=0, delay=5)

act (ports=4, param=1)
#372

act (ports=4, param=2)

act (ports=34, param=1)
act (ports=1, param=5)
#375
wait (cycle=22)
act (ports=1, param=3)
#399

act (ports=2, param=3)
act (ports=1, param=7)
#401
wait (cycle=4)
act (ports=1, param=4)
#407

wait (cycle=11069)

act (ports=8, param=4)
#11478


}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=2, step=1, delay=0)
dsu (slot=1, port=1, init_addr=2)
rep (slot=1, port=1, level=0, iter=2, step=1, delay=0)
#8

wait (cycle=11469)

act (mode=0, param=2, ports=4)
#11479

act (mode=0, param=1, ports=10)

    
}

}

}