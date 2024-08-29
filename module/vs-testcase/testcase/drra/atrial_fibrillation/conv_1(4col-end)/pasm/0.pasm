epoch <conv_1_4> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=2, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

dsu (slot=1, port=0, init_addr=2)
rep (slot=1, port=0, level=0, iter=41, step=1, delay=0)
dsu (slot=1, port=2, init_addr=2)
rep (slot=1, port=2, level=0, iter=41, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#12
wait (cycle=38)

dsu (slot=2, port=3, init_addr=0)
#52
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#54

dsu (slot=2, port=3, init_addr=2)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=0, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=5, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#61




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
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=4, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=5, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=16, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=5, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=16, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=5, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=16, step=0, delay=17)
rep (slot=4, level=1, iter=4, step=0, delay=18)
rep (slot=4, level=2, iter=5, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=4, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=5, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=5, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=18)
#37

wait (cycle=16)

act (ports=4, param=2)
#55

wait (cycle=5)

act (ports=4, param=1)
#62

act (ports=34, param=1)
act (ports=1, param=4)
#64

wait (cycle=15)

act (ports=1, param=3)
#81

wait (cycle=1133)

act (ports=8, param=3)
#1216





}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=20, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=20, step=1, delay=0)
#10

wait (cycle=1205)

act (mode=0, param=2, ports=4)
#1217

wait (cycle=4644)

act (mode=0, param=1, ports=10)





}
}

cell (x=0, y=1) {

raw {
route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=2, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

dsu (slot=1, port=0, init_addr=42)
rep (slot=1, port=0, level=0, iter=41, step=1, delay=0)
dsu (slot=1, port=2, init_addr=2)
rep (slot=1, port=2, level=0, iter=41, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#12
wait (cycle=38)

dsu (slot=2, port=3, init_addr=0)
#52
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#54

dsu (slot=2, port=3, init_addr=2)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=0, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=5, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#61



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
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=4, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=5, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=16, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=5, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=16, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=5, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=16, step=0, delay=17)
rep (slot=4, level=1, iter=4, step=0, delay=18)
rep (slot=4, level=2, iter=5, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=4, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=5, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=5, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=18)
#37

wait (cycle=16)

act (ports=4, param=2)
#55

wait (cycle=5)

act (ports=4, param=1)
#62

act (ports=34, param=1)
act (ports=1, param=4)
#64

wait (cycle=15)

act (ports=1, param=3)
#81

wait (cycle=1133)

act (ports=8, param=3)
#1216



}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=20, step=1, delay=0)
dsu (slot=1, port=1, init_addr=20)
rep (slot=1, port=1, level=0, iter=20, step=1, delay=0)
#10

wait (cycle=1205)

act (mode=0, param=2, ports=4)
#1217

wait (cycle=4644)

act (mode=0, param=1, ports=10)


    
}

}

cell (x=0, y=2) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=2, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

dsu (slot=1, port=0, init_addr=82)
rep (slot=1, port=0, level=0, iter=49, step=1, delay=0)
dsu (slot=1, port=2, init_addr=2)
rep (slot=1, port=2, level=0, iter=49, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#12
wait (cycle=46)

dsu (slot=2, port=3, init_addr=0)
#60
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#62

dsu (slot=2, port=3, init_addr=2)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=0, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=6, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#69



}
}

cell (x=1, y=2) {

raw {

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target= 6)
route (slot=0, option=0, sr=0, source=3, target= 128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=4, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=6, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=16, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=6, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=16, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=6, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=16, step=0, delay=17)
rep (slot=4, level=1, iter=4, step=0, delay=18)
rep (slot=4, level=2, iter=6, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=4, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=6, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=6, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=18)
#37

wait (cycle=24)

act (ports=4, param=2)
#63

wait (cycle=5)

act (ports=4, param=1)
#70

act (ports=34, param=1)
act (ports=1, param=4)
#72

wait (cycle=15)

act (ports=1, param=3)
#89

wait (cycle=1133)

act (ports=8, param=3)
#1224



}
}

cell (x=2, y=2) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=6, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=24, step=1, delay=0)
dsu (slot=1, port=1, init_addr=40)
rep (slot=1, port=1, level=0, iter=24, step=1, delay=0)
#10

wait (cycle=1213)

act (mode=0, param=2, ports=4)
#1225

wait (cycle=5804)

act (mode=0, param=1, ports=10)



}
}

cell (x=0, y=3) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=2, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

dsu (slot=1, port=0, init_addr=130)
rep (slot=1, port=0, level=0, iter=49, step=1, delay=0)
dsu (slot=1, port=2, init_addr=2)
rep (slot=1, port=2, level=0, iter=49, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#12
wait (cycle=46)

dsu (slot=2, port=3, init_addr=0)
#60
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#62

dsu (slot=2, port=3, init_addr=2)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=4, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=0, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=6, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#69



}
}

cell (x=1, y=3) {

raw {

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target= 6)
route (slot=0, option=0, sr=0, source=3, target= 128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=4, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=0, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=6, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=16, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=6, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=17, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=16, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=4, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=6, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=16, step=0, delay=17)
rep (slot=4, level=1, iter=4, step=0, delay=18)
rep (slot=4, level=2, iter=6, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=4, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=6, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=6, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=18)
#37

wait (cycle=24)

act (ports=4, param=2)
#63

wait (cycle=5)

act (ports=4, param=1)
#70

act (ports=34, param=1)
act (ports=1, param=4)
#72

wait (cycle=15)

act (ports=1, param=3)
#89

wait (cycle=1133)

act (ports=8, param=3)
#1224



}
}

cell (x=2, y=3) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=6, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=24, step=1, delay=0)
dsu (slot=1, port=1, init_addr=64)
rep (slot=1, port=1, level=0, iter=24, step=1, delay=0)
#10

wait (cycle=1213)

act (mode=0, param=2, ports=4)
#1225

wait (cycle=5804)

act (mode=0, param=1, ports=10)



}
}

}