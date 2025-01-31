epoch <pool_1_8> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=9, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=9, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#25

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#155






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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=15, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#153

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=68, param=1)
#156

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=31)

act (ports=8, param=3)
#192







}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#152
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#154

wait (cycle=35)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#193

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)

act (mode=0, param=1, ports=10)







}
}

cell (x=0, y=1) {

raw {
route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=10)
rep (slot=1, port=0, level=0, iter=9, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=9, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#25

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#155




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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=15, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#153

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=68, param=1)
#156

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=31)

act (ports=8, param=3)
#192




}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#151
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#153

wait (cycle=35)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#192

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=5)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)

act (mode=0, param=1, ports=10)



    
}

}

cell (x=0, y=2) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=20)
rep (slot=1, port=0, level=0, iter=9, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=9, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#24

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#154




}
}

cell (x=1, y=2) {

raw {

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=6)
route (slot=0, option=0, sr=0, source=3, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=15, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#152

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=68, param=1)
#155

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=31)

act (ports=8, param=3)
#191




}
}

cell (x=2, y=2) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#152
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#154

wait (cycle=35)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#193

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=10)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)

act (mode=0, param=1, ports=10)




}
}

cell (x=0, y=3) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=30)
rep (slot=1, port=0, level=0, iter=9, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=9, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#25

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#155




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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=15, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#153

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=68, param=1)
#156

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=31)

act (ports=8, param=3)
#192




}
}

cell (x=2, y=3) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#152
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#154

wait (cycle=35)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#193

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=15)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)

act (mode=0, param=1, ports=10)




}
}

cell (x=0, y=4) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=40)
rep (slot=1, port=0, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=11, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#25

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#155




}
}

cell (x=1, y=4) {

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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=31, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#153

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=1, delay=0)

act (ports=68, param=1)
#156

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=63)

act (ports=8, param=3)
#224




}
}

cell (x=2, y=4) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#152
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#154

wait (cycle=67)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=4)
#225

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=5, step=1, delay=0)
dsu (slot=1, port=1, init_addr=20)
rep (slot=1, port=1, level=0, iter=5, step=1, delay=0)

act (mode=0, param=1, ports=10)




}
}

cell (x=0, y=5) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=52)
rep (slot=1, port=0, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=11, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#25

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#155



}
}

cell (x=1, y=5) {

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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=31, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#153

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=1, delay=0)

act (ports=68, param=1)
#156

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=63)

act (ports=8, param=3)
#224



}
}

cell (x=2, y=5) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#152
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#154

wait (cycle=67)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=4)
#225

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=5, step=1, delay=0)
dsu (slot=1, port=1, init_addr=26)
rep (slot=1, port=1, level=0, iter=5, step=1, delay=0)

act (mode=0, param=1, ports=10)



}
}

cell (x=0, y=6) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=64)
rep (slot=1, port=0, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=11, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#25

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#155



}
}

cell (x=1, y=6) {

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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=31, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#153

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=1, delay=0)

act (ports=68, param=1)
#156

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=63)

act (ports=8, param=3)
#224



}
}

cell (x=2, y=6) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#152
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#154

wait (cycle=67)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=4)
#225

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=5, step=1, delay=0)
dsu (slot=1, port=1, init_addr=32)
rep (slot=1, port=1, level=0, iter=5, step=1, delay=0)

act (mode=0, param=1, ports=10)



}
}

cell (x=0, y=7) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=76)
rep (slot=1, port=0, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=11, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=1, step=4, delay=60)

act (mode=0, param=1, ports=0b0101)
#10

wait (cycle=13)

act (mode=0, param=2, ports=0b1000)
#25

wait (cycle=126)

dsu (slot=2, port=3, init_addr=8)
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#155



}
}

cell (x=1, y=7) {

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
rep (slot=1, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=1, step=0, delay=60)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=1, step=0, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=1, step=0, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=-1, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=0, step=0, delay=0)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#25

act (ports=68, param=1)
#26

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#29

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=31, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=31, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=31, step=1, delay=1)
#40

wait (cycle=111)

act (ports=8, param=3)
#153

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=1, delay=0)

act (ports=68, param=1)
#156

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)

wait (cycle=63)

act (ports=8, param=3)
#224



}
}

cell (x=2, y=7) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=148)

dsu (slot=2, port=2, init_addr=0)
#152
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#154

wait (cycle=67)

dsu (slot=2, port=2, init_addr=4)
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=4)
#225

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=5, step=1, delay=0)
dsu (slot=1, port=1, init_addr=38)
rep (slot=1, port=1, level=0, iter=5, step=1, delay=0)

act (mode=0, param=1, ports=10)



}
}

}