epoch <conv_1_2> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=42, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=42, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)
#9

wait (cycle=26)

act (mode=0, param=2, ports=0b1000)
#37

dsu (slot=2, port=3, init_addr=2)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=4, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#44

wait (cycle=5787)

dsu (slot=1, port=0, init_addr=42)
#5833
rep (slot=1, port=0, level=0, iter=48, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=48, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=5, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=1, ports=0b0101)

act (mode=0, param=2, ports=0b1000)
#5844



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
rep (slot=1, port=2, level=1, iter=3, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=4, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=-1, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=4, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=4, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=17)
rep (slot=4, level=1, iter=3, step=0, delay=18)
rep (slot=4, level=2, iter=4, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=4, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=4, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=18)

act (ports=4, param=2)
#38

wait (cycle=5)

act (ports=4, param=1)
#45

act (ports=34, param=1)
act (ports=1, param=4)
#47

wait (cycle=15)

act (ports=1, param=3)
#64

wait (cycle=1133)

act (ports=8, param=3)
#1199

wait (cycle=4616)

dsu (slot=1, port=2, init_addr=0)
#5817
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=5, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=-1, step=0, delay=4)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=5, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=5, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=17)
rep (slot=4, level=1, iter=3, step=0, delay=18)
rep (slot=4, level=2, iter=5, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=5, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=5, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=18)

act (ports=4, param=1)
#5845

act (ports=34, param=1)
act (ports=1, param=4)
#5847

wait (cycle=15)

act (ports=1, param=3)
#5864

wait (cycle=1133)

act (ports=8, param=3)
#6999




}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=4, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=19, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=19, step=1, delay=0)
#10

wait (cycle=1188)

act (mode=0, param=2, ports=4)
#1200

wait (cycle=4644)

act (mode=0, param=1, ports=10)
#5846

wait (cycle=1144)

dsu (slot=2, port=2, init_addr=0)
#6992
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=23, step=1, delay=0)
dsu (slot=1, port=1, init_addr=20)
rep (slot=1, port=1, level=0, iter=23, step=1, delay=0)

act (mode=0, param=2, ports=4)
#7000

wait (cycle=5804)

act (mode=0, param=1, ports=10)




}
}

cell (x=0, y=1) {

raw {
route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

dsu (slot=1, port=0, init_addr=90)
rep (slot=1, port=0, level=0, iter=40, step=1, delay=0)
dsu (slot=1, port=2, init_addr=2)
rep (slot=1, port=2, level=0, iter=40, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)
#14

wait (cycle=21)

act (mode=0, param=2, ports=0b1000)
#37

dsu (slot=2, port=3, init_addr=2)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=4, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#44

wait (cycle=5787)

dsu (slot=1, port=0, init_addr=130)
#5833
rep (slot=1, port=0, level=0, iter=48, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=48, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=5, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=1, ports=0b0101)

act (mode=0, param=2, ports=0b1000)
#5844


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
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=4, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=-1, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=4, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=4, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=17)
rep (slot=4, level=1, iter=3, step=0, delay=18)
rep (slot=4, level=2, iter=4, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=4, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=4, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=18)

act (ports=4, param=2)
#38

wait (cycle=5)

act (ports=4, param=1)
#45

act (ports=34, param=1)
act (ports=1, param=4)
#47

wait (cycle=15)

act (ports=1, param=3)
#64

wait (cycle=1133)

act (ports=8, param=3)
#1199

wait (cycle=4616)

dsu (slot=1, port=2, init_addr=0)
#5817
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=5, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=-1, step=0, delay=4)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=1, port=1, level=3, iter=5, step=0, delay=6)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=2)
rep (slot=2, port=1, level=3, iter=5, step=0, delay=6)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=17)
rep (slot=4, level=1, iter=3, step=0, delay=18)
rep (slot=4, level=2, iter=5, step=0, delay=22)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=17)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)
rep (slot=3, port=0, level=2, iter=5, step=0, delay=22)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=5, step=0, delay=4)
repx (slot=3, port=3, level=1, iter=-1, step=0, delay=18)

act (ports=4, param=1)
#5845

act (ports=34, param=1)
act (ports=1, param=4)
#5847

wait (cycle=15)

act (ports=1, param=3)
#5864

wait (cycle=1133)

act (ports=8, param=3)
#6999


}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=4, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=19, step=1, delay=0)
dsu (slot=1, port=1, init_addr=44)
rep (slot=1, port=1, level=0, iter=19, step=1, delay=0)
#10

wait (cycle=1188)

act (mode=0, param=2, ports=4)
#1200

wait (cycle=4644)

act (mode=0, param=1, ports=10)
#5846

wait (cycle=1144)

dsu (slot=2, port=2, init_addr=0)
#6992
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=23, step=1, delay=0)
dsu (slot=1, port=1, init_addr=64)
rep (slot=1, port=1, level=0, iter=23, step=1, delay=0)

act (mode=0, param=2, ports=4)
#7000

wait (cycle=5804)

act (mode=0, param=1, ports=10)

    
}

}

}