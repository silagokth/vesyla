epoch <rb0> {
    cell (x=0, y=0){
        raw{
            route (slot=0, option=0, sr=0, source=2, target= 128)
            act (mode=0, param=0, ports=0b0100)

            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, step=1, iter=16, delay=0)

            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, step=1, iter=16, delay=0)

            dsu (slot=2, port=3, init_addr=16)
            rep (slot=2, port=3, level=0, iter=0, step=1, delay=0)

            act (mode=0, param=1, ports=0b0101)
            wait (cycle = 24)
            act (mode=0, param=2, ports=0b1000)
            # io -> rf
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=3, step=4, delay=60)
            act (mode=0, param=2, ports=0b1000)
            halt
        }
    }


    cell (x=1, y=0){
        raw{
            route (slot=0, option=0, sr=1, source=1, target= 0b001010)
            route (slot=0, option=0, sr=0, source=2, target= 128)
            route (slot=0, option=1, sr=0, source=3, target= 128)
            fsm (slot=0, port=2, delay_0=10)
            rep (slot=0, port=2, level=0, iter=4, delay=52)

            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
            rep (slot=1, port=2, level=1, iter=3, step=0, delay=60)

            dsu (slot=3, port=2, init_addr=0)
            rep (slot=3, port=2, level=0, iter=0, step=1, delay=0)

            swb (slot=0, option=0, source=1, target=4)
            swb (slot=0, option=0, source=3, target=5)
            swb (slot=0, option=0, source=4, target=6)
            swb (slot=0, option=0, source=6, target=2)
            swb (slot=0, option=0, source=2, target=8)
            swb (slot=0, option=0, source=8, target=3)


            #to-dpu
            dsu (slot=3, port=1, init_addr=0)
            rep (slot=3, port=1, level=0, iter=-1, step=0, delay=0)
            repx (slot=3, port=1, level=0, iter=3, step=0, delay=0)

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)

            dpu (slot=4, option=0, mode=4)

            dpu (slot=6, option=0, mode=6, immediate=2)

            dsu (slot=2, port=0, init_addr=0)
            rep (slot=2, port=0, level=0, iter=-1, step=1, delay=0)
            repx (slot=2, port=0, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=0, level=1, iter=3, step=0, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)

            dpu (slot=8, option=0, mode=2)
            act (mode=0, param=0, ports=0b0100)

            act (mode=0, ports=4, param=3)
            act (ports=1)
            wait (cycle = 1)
            act (mode=0, ports=4, param=1)

            act (ports=0b001000000010, param=1)
            act (ports=1, param=4)
            act (ports=1, param=6)
            act (ports=1, param=2)
            act (ports=2, param=2)
            act (ports=1, param=8)

            wait (cycle = 54)

            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=3, step=0, delay=60)
            act (ports=0b1000, param=2)


            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=0, step=1, delay=0)

            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=0, step=0, delay=0)

            wait (cycle = 191)
            act (ports=1, param=3)



            act (ports=0b1000, param=3)

            halt
        }
    }

    cell (x=2, y=0){
        raw{
            route (slot=0, option=0, sr=1, source=1, target= 0b0000000000000100)

            wait (cycle = 93)

            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
            rep (slot=2, port=2, level=1, iter=3, step=4, delay=60)

            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=3, step=1, delay=0)
            rep (slot=1, port=3, level=1, iter=3, step=4, delay=60)
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=3, step=1, delay=0)
            rep (slot=1, port=1, level=1, iter=3, step=4, delay=60)


            act (mode=0, param=0, ports=0b0100)
            act (ports=0b0100, param=2)
            act (ports=0b1010, param=1)

            wait (cycle = 193)

            dsu (slot=2, port=2, init_addr=16)
            rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)

            act (ports=0b0100, param=2)

            dsu (slot=1, port=3, init_addr=16)
            rep (slot=1, port=3, level=0, iter=0, step=1, delay=0)
            dsu (slot=1, port=1, init_addr=16)
            rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)

            act (ports=0b1010, param=1)
            halt

        }
    }
}
