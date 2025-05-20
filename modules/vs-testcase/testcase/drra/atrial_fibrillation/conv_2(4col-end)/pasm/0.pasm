epoch <conv_2_4> {


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
#9
wait (cycle=63)

dsu (slot=2, port=3, init_addr=63)
#74
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#76

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=0)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#81

wait (cycle=1034)

dsu (slot=2, port=3, init_addr=8)
#1117
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#1119





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

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=4)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=4)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
rep (slot=4, level=1, iter=3, step=0, delay=18)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#29

wait (cycle=46)

act (ports=4, param=2)
#77

wait (cycle=3)

act (ports=4, param=1)
#82

act (ports=34, param=1)
act (ports=1, param=4)
#84
wait (cycle=13)
act (ports=1, param=3)
#99

wait (cycle=1001)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)

act (ports=8, param=3)
#1114

wait (cycle=2)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=4, param=1)
#1120

act (ports=34, param=1)
act (ports=1, param=4)
#1122
wait (cycle=13)
act (ports=1, param=3)
#1137

wait (cycle=240)

act (ports=8, param=3)
#1379






}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
#8

wait (cycle=1105)

act (mode=0, param=2, ports=4)
#1115

wait (cycle=261)

dsu (slot=2, port=2, init_addr=4)
#1378
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#1380

act (mode=0, param=1, ports=10)






}
}

cell (x=0, y=1) {

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
#9
wait (cycle=63)

dsu (slot=2, port=3, init_addr=63)
#74
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#76

dsu (slot=2, port=3, init_addr=10)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=0)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#81

wait (cycle=1034)

dsu (slot=2, port=3, init_addr=18)
#1117
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#1119




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
rep (slot=1, port=2, level=1, iter=3, step=0, delay=0)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=4)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=4)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
rep (slot=4, level=1, iter=3, step=0, delay=18)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#29

wait (cycle=46)

act (ports=4, param=2)
#77

wait (cycle=3)

act (ports=4, param=1)
#82

act (ports=34, param=1)
act (ports=1, param=4)
#84
wait (cycle=13)
act (ports=1, param=3)
#99

wait (cycle=1001)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)

act (ports=8, param=3)
#1114

wait (cycle=2)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=4, param=1)
#1120

act (ports=34, param=1)
act (ports=1, param=4)
#1122
wait (cycle=13)
act (ports=1, param=3)
#1137

wait (cycle=240)

act (ports=8, param=3)
#1379




}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=5)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
#8

wait (cycle=1105)

act (mode=0, param=2, ports=4)
#1115

wait (cycle=261)

dsu (slot=2, port=2, init_addr=4)
#1378
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#1380

act (mode=0, param=1, ports=10)



    
}

}

cell (x=0, y=2) {

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
#9
wait (cycle=63)

dsu (slot=2, port=3, init_addr=63)
#74
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#76

dsu (slot=2, port=3, init_addr=20)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=0)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#81

wait (cycle=1034)

dsu (slot=2, port=3, init_addr=28)
#1117
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#1119




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
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=0)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=4)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=4)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
rep (slot=4, level=1, iter=3, step=0, delay=18)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#29

wait (cycle=46)

act (ports=4, param=2)
#77

wait (cycle=3)

act (ports=4, param=1)
#82

act (ports=34, param=1)
act (ports=1, param=4)
#84
wait (cycle=13)
act (ports=1, param=3)
#99

wait (cycle=1001)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)

act (ports=8, param=3)
#1114

wait (cycle=2)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=4, param=1)
#1120

act (ports=34, param=1)
act (ports=1, param=4)
#1122
wait (cycle=13)
act (ports=1, param=3)
#1137

wait (cycle=240)

act (ports=8, param=3)
#1379




}
}

cell (x=2, y=2) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
dsu (slot=1, port=1, init_addr=10)
rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
#8

wait (cycle=1105)

act (mode=0, param=2, ports=4)
#1115

wait (cycle=261)

dsu (slot=2, port=2, init_addr=4)
#1378
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#1380

act (mode=0, param=1, ports=10)




}
}

cell (x=0, y=3) {

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
#9
wait (cycle=63)

dsu (slot=2, port=3, init_addr=63)
#74
rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#76

dsu (slot=2, port=3, init_addr=30)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=0)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#81

wait (cycle=1034)

dsu (slot=2, port=3, init_addr=38)
#1117
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#1119

wait (cycle=258)

dsu (slot=2, port=3, init_addr=40)
#1379
rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#1381




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
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=0)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)
rep (slot=1, port=1, level=2, iter=3, step=0, delay=4)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=3, step=0, delay=4)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
rep (slot=4, level=1, iter=3, step=0, delay=18)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)
rep (slot=3, port=0, level=1, iter=3, step=16, delay=18)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
#29

wait (cycle=46)

act (ports=4, param=2)
#77

wait (cycle=3)

act (ports=4, param=1)
#82

act (ports=34, param=1)
act (ports=1, param=4)
#84
wait (cycle=13)
act (ports=1, param=3)
#99

wait (cycle=1001)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=1, port=1, level=1, iter=15, step=2, delay=1)

dsu (slot=2, port=1, init_addr=0)
rep (slot=2, port=1, level=0, iter=14, step=1, delay=0)
rep (slot=2, port=1, level=1, iter=15, step=0, delay=1)

dpu (slot=4, option=0, mode=8)
rep (slot=4, level=0, iter=15, step=0, delay=15)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=15, step=1, delay=15)

act (ports=8, param=3)
#1114

wait (cycle=2)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)

act (ports=4, param=1)
#1120

act (ports=34, param=1)
act (ports=1, param=4)
#1122
wait (cycle=13)
act (ports=1, param=3)
#1137

wait (cycle=228)

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

act (ports=8, param=3)
#1379

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=1, step=1, delay=0)

act (ports=4, param=1)
#1382

act (ports=34, param=1)
act (ports=1, param=4)
#1384
wait (cycle=13)
act (ports=1, param=3)
#1399

wait (cycle=385)

act (ports=8, param=3)
#1786




}
}

cell (x=2, y=3) {

raw {

route (slot=0, option=0, sr=1, source=1, target= 4)

act (mode=0, param=0, ports=0b0100)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=6, step=1, delay=0)
dsu (slot=1, port=1, init_addr=15)
rep (slot=1, port=1, level=0, iter=6, step=1, delay=0)
#8

wait (cycle=1105)

act (mode=0, param=2, ports=4)
#1115

wait (cycle=261)

dsu (slot=2, port=2, init_addr=4)
#1378
rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

act (mode=0, param=2, ports=4)
#1380

wait (cycle=403)

dsu (slot=2, port=2, init_addr=5)
#1785
rep (slot=2, port=2, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=4)
#1787

act (mode=0, param=1, ports=10)




}
}

}