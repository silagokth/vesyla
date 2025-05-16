epoch <conv_1_1> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=42, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=42, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#7
wait (cycle=42)

dsu (slot=2, port=3, init_addr=0)
#51
rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#53

dsu (slot=2, port=3, init_addr=2)
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=4, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#60

wait (cycle=5746)

dsu (slot=1, port=0, init_addr=42)
#5808
rep (slot=1, port=0, level=0, iter=40, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=40, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#5812

wait (cycle=40)

dsu (slot=2, port=3, init_addr=0)
#5854
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=4, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#5860

wait (cycle=5743)

dsu (slot=1, port=0, init_addr=82)
#11605
rep (slot=1, port=0, level=0, iter=48, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=48, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#11609

wait (cycle=42)

dsu (slot=2, port=3, init_addr=0)
#11653
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=5, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#11659

wait (cycle=6904)

dsu (slot=1, port=0, init_addr=130)
#18565
rep (slot=1, port=0, level=0, iter=48, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=48, step=1, delay=0)

act (mode=0, param=1, ports=0b0101)
#18569

wait (cycle=42)

dsu (slot=2, port=3, init_addr=0)
#18613
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=3, step=2, delay=30)
repx (slot=2, port=3, level=1, iter=-1, step=0, delay=4)
rep (slot=2, port=3, level=2, iter=5, step=8, delay=34)
repx (slot=2, port=3, level=2, iter=-1, step=0, delay=4)

act (mode=0, param=2, ports=0b1000)
#18619


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
#37

wait (cycle=15)

act (ports=4, param=2)
#54

wait (cycle=5)

act (ports=4, param=1)
#61

act (ports=34, param=1)
act (ports=1, param=4)
#63

wait (cycle=15)

act (ports=1, param=3)
#80

wait (cycle=1133)

act (ports=8, param=3)
#1215

wait (cycle=4616)

dsu (slot=1, port=2, init_addr=0)
#5833
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=3, step=0, delay=30)
repx (slot=1, port=2, level=1, iter=-1, step=0, delay=4)
rep (slot=1, port=2, level=2, iter=4, step=0, delay=34)
repx (slot=1, port=2, level=2, iter=-1, step=0, delay=4)

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

act (ports=4, param=1)
#5861

act (ports=34, param=1)
act (ports=1, param=4)
#5863

wait (cycle=15)

act (ports=1, param=3)
#5880

wait (cycle=1133)

act (ports=8, param=3)
#7015

wait (cycle=4615)

dsu (slot=1, port=2, init_addr=0)
#11632
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
#11660

act (ports=34, param=1)
act (ports=1, param=4)
#11662

wait (cycle=15)

act (ports=1, param=3)
#11679

wait (cycle=1133)

act (ports=8, param=3)
#12814

wait (cycle=5776)

dsu (slot=1, port=2, init_addr=0)
#18592
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
#18620

act (ports=34, param=1)
act (ports=1, param=4)
#18622

wait (cycle=15)

act (ports=1, param=3)
#18639

wait (cycle=1133)

act (ports=8, param=3)
#19774



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

wait (cycle=1204)

act (mode=0, param=2, ports=4)
#1216

wait (cycle=4643)

act (mode=0, param=1, ports=10)
#5861

wait (cycle=1145)

dsu (slot=2, port=2, init_addr=0)
#7008
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=4, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=19, step=1, delay=0)
dsu (slot=1, port=1, init_addr=20)
rep (slot=1, port=1, level=0, iter=19, step=1, delay=0)

act (mode=0, param=2, ports=4)
#7016

wait (cycle=4644)

act (mode=0, param=1, ports=10)
#11662

wait (cycle=1143)

dsu (slot=2, port=2, init_addr=0)
#12807
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=23, step=1, delay=0)
dsu (slot=1, port=1, init_addr=40)
rep (slot=1, port=1, level=0, iter=23, step=1, delay=0)

act (mode=0, param=2, ports=4)
#12815

wait (cycle=5804)

act (mode=0, param=1, ports=10)
#18621

wait (cycle=1144)

dsu (slot=2, port=2, init_addr=0)
#19767
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=5, step=4, delay=4)
repx (slot=2, port=2, level=1, iter=-1, step=0, delay=18)

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=23, step=1, delay=0)
dsu (slot=1, port=1, init_addr=64)
rep (slot=1, port=1, level=0, iter=23, step=1, delay=0)

act (mode=0, param=2, ports=4)
#19775

wait (cycle=5804)

act (mode=0, param=1, ports=10)



}
}

}