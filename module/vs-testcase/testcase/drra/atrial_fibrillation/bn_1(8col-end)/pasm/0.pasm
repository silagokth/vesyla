epoch <bn_1_8> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202





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

route (slot=0, option=0, sr=1, source=1, target=22)
route (slot=0, option=0, sr=0, source=9, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258






}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)






}
}

cell (x=0, y=1) {

raw {
route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=11)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202



}

}

cell (x=1, y=1) {

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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258



}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=11)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)


    
}

}

cell (x=0, y=2) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=22)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202



}
}

cell (x=1, y=2) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=4, target=8)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=7, target=9)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=22)
route (slot=0, option=0, sr=0, source=9, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258



}
}

cell (x=2, y=2) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=22)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)



}
}

cell (x=0, y=3) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=33)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202



}
}

cell (x=1, y=3) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=4, target=8)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=7, target=9)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=22)
route (slot=0, option=0, sr=0, source=9, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258



}
}

cell (x=2, y=3) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=33)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)



}
}

cell (x=0, y=4) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=46)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202


}
}

cell (x=1, y=4) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=4, target=8)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=7, target=9)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=22)
route (slot=0, option=0, sr=0, source=9, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258


}
}

cell (x=2, y=4) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=44)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)


}
}

cell (x=0, y=5) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=57)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202


}
}

cell (x=1, y=5) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=4, target=8)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=7, target=9)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=22)
route (slot=0, option=0, sr=0, source=9, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258


}
}

cell (x=2, y=5) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=55)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)


}
}

cell (x=0, y=6) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=68)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202


}
}

cell (x=1, y=6) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=4, target=8)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=7, target=9)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=22)
route (slot=0, option=0, sr=0, source=9, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258


}
}

cell (x=2, y=6) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=66)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)


}
}

cell (x=0, y=7) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=79)
rep (slot=1, port=0, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7

wait (cycle=10)

dsu (slot=1, port=0, init_addr=44)
#19
rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
dsu (slot=1, port=2, init_addr=11)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#23

wait (cycle=21)

dsu (slot=2, port=3, init_addr=11)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#48

dsu (slot=2, port=3, init_addr=12)
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#51

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=1)

act (mode=0, param=2, ports=0b1000)
#56

wait (cycle=142)

dsu (slot=2, port=3, init_addr=8)
#200
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#202


}
}

cell (x=1, y=7) {

raw {

swb (slot=0, option=0, source=1, target=5)
swb (slot=0, option=0, source=2, target=6)
swb (slot=0, option=0, source=3, target=7)
swb (slot=0, option=0, source=4, target=8)
swb (slot=0, option=0, source=5, target=3)
swb (slot=0, option=0, source=7, target=9)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=22)
route (slot=0, option=0, sr=0, source=9, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=5)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=4, port=2, init_addr=0)
rep (slot=4, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=-1, step=1, delay=0)
repx (slot=3, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=3, port=1, level=1, iter=1, step=0, delay=9)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=-1, step=0, delay=0)
repx (slot=4, port=1, level=0, iter=0, step=0, delay=0)
rep (slot=4, port=1, level=1, iter=1, step=0, delay=9)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=-1, step=1, delay=0)
repx (slot=9, port=0, level=0, iter=0, step=0, delay=0)
rep (slot=9, port=0, level=1, iter=1, step=0, delay=9)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=9, port=3, level=1, iter=1, step=0, delay=5)
repx (slot=9, port=3, level=1, iter=-1, step=0, delay=1)
#48

act (ports=4, param=2)

wait (cycle=1)

act (ports=4, param=4)
#52

wait (cycle=3)

act (ports=4, param=1)
#57

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#63

wait (cycle=59)

act (ports=8, param=9)
#124

wait (cycle=59)

dsu (slot=1, port=2, init_addr=0)
#185
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=5, option=0, mode=7)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=3, port=1, init_addr=0)
rep (slot=3, port=1, level=0, iter=47, step=1, delay=0)

dsu (slot=4, port=1, init_addr=0)
rep (slot=4, port=1, level=0, iter=47, step=0, delay=0)

dpu (slot=7, option=0, mode=1)
dsu (slot=9, port=0, init_addr=0)
rep (slot=9, port=0, level=0, iter=47, step=1, delay=0)

dsu (slot=9, port=3, init_addr=0)
rep (slot=9, port=3, level=0, iter=2, step=1, delay=0)

act (ports=4, param=1)
#203

act (ports=34, param=1)
act (ports=1, param=5)
act (ports=1, param=3)
act (ports=34, param=3)
act (ports=1, param=7)
act (ports=1, param=9)
#209

wait (cycle=47)

act (ports=8, param=9)
#258


}
}

cell (x=2, y=7) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=4, delay=5)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=1)

wait (cycle=117)

act (mode=0, param=2, ports=4)
#125

wait (cycle=130)

dsu (slot=2, port=2, init_addr=8)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=4)
#259

wait (cycle=4)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=10, step=1, delay=0)
dsu (slot=1, port=1, init_addr=77)
rep (slot=1, port=1, level=0, iter=10, step=1, delay=0)

act (mode=0, param=1, ports=10)


}
}

}