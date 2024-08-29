epoch <pool_1_4> {


cell (x=0, y=0) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=0)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=2, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)
#12

wait (cycle=21)

act (mode=0, param=2, ports=0b1000)
#35

wait (cycle=262)

dsu (slot=2, port=3, init_addr=16)
#299
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#301

wait (cycle=60)

dsu (slot=2, port=3, init_addr=20)
#363
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#365






}

}

cell (x=1, y=0) {

raw {

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=6)
route (slot=0, option=0, sr=0, source=3, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=1, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=2, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=2, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=2, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=2, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)
#34

wait (cycle=0)

act (ports=68, param=1)
#36

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#39

wait (cycle=122)

act (ports=8, param=3)
#163

wait (cycle=126)

dsu (slot=1, port=2, init_addr=0)
#291
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=32, step=1, delay=1)

act (ports=68, param=1)
#302

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#305

wait (cycle=46)

dsu (slot=1, port=2, init_addr=0)
#353
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=16, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=32)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=1)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)

act (ports=68, param=1)
#366

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#369

wait (cycle=27)

act (ports=8, param=3)
#398







}
}

cell (x=2, y=0) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=156)

dsu (slot=2, port=2, init_addr=0)
#160
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

act (mode=0, param=2, ports=4)
#164

wait (cycle=231)

dsu (slot=2, port=2, init_addr=8)
#397
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#399

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=11, step=1, delay=0)

act (mode=0, param=1, ports=10)







}
}

cell (x=0, y=1) {

raw {
route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=22)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=2, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)
#12

wait (cycle=21)

act (mode=0, param=2, ports=0b1000)
#35

wait (cycle=262)

dsu (slot=2, port=3, init_addr=16)
#299
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#301

wait (cycle=60)

dsu (slot=2, port=3, init_addr=20)
#363
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#365





}

}

cell (x=1, y=1) {

raw {
swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=6)
route (slot=0, option=0, sr=0, source=3, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=1, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=2, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=2, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=2, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=2, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)
#34

wait (cycle=0)

act (ports=68, param=1)
#36

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#39

wait (cycle=122)

act (ports=8, param=3)
#163

wait (cycle=126)

dsu (slot=1, port=2, init_addr=0)
#291
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=32, step=1, delay=1)

act (ports=68, param=1)
#302

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#305

wait (cycle=46)

dsu (slot=1, port=2, init_addr=0)
#353
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=16, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=32)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=1)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)

act (ports=68, param=1)
#366

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#369

wait (cycle=27)

act (ports=8, param=3)
#398





}

}

cell (x=2, y=1) {

raw {
route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=156)

dsu (slot=2, port=2, init_addr=0)
#160
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

act (mode=0, param=2, ports=4)
#164

wait (cycle=231)

dsu (slot=2, port=2, init_addr=8)
#397
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#399

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=1, init_addr=11)
rep (slot=1, port=1, level=0, iter=11, step=1, delay=0)

act (mode=0, param=1, ports=10)




    
}

}

cell (x=0, y=2) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=44)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=2, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)
#12

wait (cycle=21)

act (mode=0, param=2, ports=0b1000)
#35

wait (cycle=262)

dsu (slot=2, port=3, init_addr=16)
#299
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#301

wait (cycle=60)

dsu (slot=2, port=3, init_addr=20)
#363
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#365





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
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=1, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=2, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=2, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=2, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=2, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)
#34

wait (cycle=0)

act (ports=68, param=1)
#36

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#39

wait (cycle=122)

act (ports=8, param=3)
#163

wait (cycle=126)

dsu (slot=1, port=2, init_addr=0)
#291
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=32, step=1, delay=1)

act (ports=68, param=1)
#302

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#305

wait (cycle=46)

dsu (slot=1, port=2, init_addr=0)
#353
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=16, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=32)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=1)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)

act (ports=68, param=1)
#366

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#369

wait (cycle=27)

act (ports=8, param=3)
#398





}
}

cell (x=2, y=2) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=156)

dsu (slot=2, port=2, init_addr=0)
#160
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

act (mode=0, param=2, ports=4)
#164

wait (cycle=231)

dsu (slot=2, port=2, init_addr=8)
#397
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#399

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=1, init_addr=22)
rep (slot=1, port=1, level=0, iter=11, step=1, delay=0)

act (mode=0, param=1, ports=10)





}
}

cell (x=0, y=3) {

raw {

route (slot=0, option=0, sr=0, source=2, target=0b010000000)
act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=0, init_addr=66)
rep (slot=1, port=0, level=0, iter=22, step=1, delay=0)
dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=22, step=1, delay=0)

dsu (slot=2, port=3, init_addr=0)
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=3, level=1, iter=2, step=4, delay=60)
rep (slot=2, port=3, level=2, iter=2, step=8, delay=1)
repx (slot=2, port=3, level=2, iter=0, step=0, delay=1)

act (mode=0, param=1, ports=0b0101)
#12

wait (cycle=21)

act (mode=0, param=2, ports=0b1000)
#35

wait (cycle=262)

dsu (slot=2, port=3, init_addr=16)
#299
rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#301

wait (cycle=60)

dsu (slot=2, port=3, init_addr=20)
#363
rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)

act (mode=0, param=2, ports=0b1000)
#365





}
}

cell (x=1, y=3) {

raw {

swb (slot=0, option=0, source=1, target=4)
swb (slot=0, option=0, source=2, target=5)
swb (slot=0, option=0, source=4, target=3)

act (mode=0, param=0, ports=0b0001)

route (slot=0, option=0, sr=1, source=1, target=6)
route (slot=0, option=0, sr=0, source=3, target=128)

act (mode=0, param=0, ports=0b0100)

dsu (slot=1, port=2, init_addr=0)
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=1, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=1, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=1, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=0, delay=60)
rep (slot=2, port=2, level=2, iter=2, step=0, delay=1)
repx (slot=2, port=2, level=2, iter=0, step=0, delay=1)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=1, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=1, port=1, level=2, iter=2, step=0, delay=6)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)
rep (slot=2, port=1, level=1, iter=2, step=0, delay=1)
rep (slot=2, port=1, level=2, iter=2, step=0, delay=6)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=0, step=1, delay=1)
repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
rep (slot=3, port=0, level=1, iter=2, step=0, delay=6)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=4, step=1, delay=0)
rep (slot=3, port=3, level=1, iter=2, step=0, delay=1)
repx (slot=3, port=3, level=1, iter=0, step=0, delay=2)
#34

wait (cycle=0)

act (ports=68, param=1)
#36

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#39

wait (cycle=122)

act (ports=8, param=3)
#163

wait (cycle=126)

dsu (slot=1, port=2, init_addr=0)
#291
rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=32, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=32, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=0)
rep (slot=3, port=0, level=0, iter=32, step=1, delay=1)

act (ports=68, param=1)
#302

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#305

wait (cycle=46)

dsu (slot=1, port=2, init_addr=0)
#353
rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=2, port=2, init_addr=0)
rep (slot=2, port=2, level=0, iter=2, step=1, delay=0)

dsu (slot=1, port=1, init_addr=0)
rep (slot=1, port=1, level=0, iter=16, step=2, delay=1)

dsu (slot=2, port=1, init_addr=1)
rep (slot=2, port=1, level=0, iter=16, step=2, delay=1)

dpu (slot=4, option=0, mode=16)
dsu (slot=3, port=0, init_addr=32)
rep (slot=3, port=0, level=0, iter=16, step=1, delay=1)

dsu (slot=3, port=3, init_addr=0)
rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)

act (ports=68, param=1)
#366

act (ports=34, param=1)
act (ports=1, param=4)
act (ports=1, param=3)
#369

wait (cycle=27)

act (ports=8, param=3)
#398





}
}

cell (x=2, y=3) {

raw {

route (slot=0, option=0, sr=1, source=1, target=4)

act (mode=0, param=0, ports=0b0100)

wait (cycle=156)

dsu (slot=2, port=2, init_addr=0)
#160
rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
rep (slot=2, port=2, level=1, iter=2, step=4, delay=1)
repx (slot=2, port=2, level=1, iter=0, step=0, delay=2)

act (mode=0, param=2, ports=4)
#164

wait (cycle=231)

dsu (slot=2, port=2, init_addr=8)
#397
rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)

act (mode=0, param=2, ports=4)
#399

dsu (slot=1, port=3, init_addr=0)
rep (slot=1, port=3, level=0, iter=11, step=1, delay=0)
dsu (slot=1, port=1, init_addr=33)
rep (slot=1, port=1, level=0, iter=11, step=1, delay=0)

act (mode=0, param=1, ports=10)





}
}

}