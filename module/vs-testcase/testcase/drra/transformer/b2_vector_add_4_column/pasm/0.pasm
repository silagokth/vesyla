epoch <rb0> {
    cell (x=0, y=0){
        raw{
           # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=2, step=4, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=2, step=4, delay=0)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=2, step=4, delay=0)

            wait (cycle=0)
            act (mode=0, param=2, ports=0b1000)
            halt
        }
    }

    cell (x=1, y=0){
        raw{
          # build incoming route

            route (slot=0, option=0, sr=1, source=1, target=0b0110)
            route (slot=0, option=0, sr=0, source=3, target=128)
            act (mode=0, param=0, ports=0b0100)

            # connect RF1 to DPU4, RF2 to DPU5, DPU4 acc to RF3
            swb (slot=0, option=0, source=1, target=4)
            swb (slot=0, option=0, source=2, target=5)
            swb (slot=0, option=0, source=4, target=3)
            act (ports=1)

            # write to RF1
            dpu (slot=4, option=0, mode=1)
            act (ports=1, param=4)

            dsu (slot=1, port=2, init_addr=0)
            dsu (slot=2, port=2, init_addr=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu

            # read RF1 64 elements at a time, repeat 64 times, delay till RF2 reloaded 

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)



            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=16, step=1, delay=0)

            act (mode=0, param=1, ports=34)

            wait (cycle=0)
            act (mode=0, param=3, ports=0b0001)

            # read bulk RF3
            dsu (slot=3, port=3, init_addr=0)



            wait (cycle=13)
            act (ports=0b1000, param=3)

            halt
        }

    }

    cell (x=2, y=0){
        raw{
          # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=34)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            dsu (slot=1, port=1, init_addr=0)

            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=1){
        raw{
            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data
            dsu (slot=1, port=0, init_addr=1)
            rep (slot=1, port=0, level=0, iter=2, step=4, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=2, step=4, delay=0)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=2, step=4, delay=0)

            wait (cycle=0)
            act (mode=0, param=2, ports=0b1000)
            halt
        }
    }

    cell (x=1, y=1){
        raw{
            # build incoming route

            route (slot=0, option=0, sr=1, source=1, target=0b0110)
            route (slot=0, option=0, sr=0, source=3, target=128)
            act (mode=0, param=0, ports=0b0100)

            # connect RF1 to DPU4, RF2 to DPU5, DPU4 acc to RF3
            swb (slot=0, option=0, source=1, target=4)
            swb (slot=0, option=0, source=2, target=5)
            swb (slot=0, option=0, source=4, target=3)
            act (ports=1)

            # write to RF1
            dpu (slot=4, option=0, mode=1)
            act (ports=1, param=4)

            dsu (slot=1, port=2, init_addr=0)
            dsu (slot=2, port=2, init_addr=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu

            # read RF1 64 elements at a time, repeat 64 times, delay till RF2 reloaded 

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)



            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=16, step=1, delay=0)

            act (mode=0, param=1, ports=34)

            wait (cycle=0)
            act (mode=0, param=3, ports=0b0001)

            # read bulk RF3
            dsu (slot=3, port=3, init_addr=0)



            wait (cycle=13)
            act (ports=0b1000, param=3)

            halt
        }
    }

    cell (x=2, y=1){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=34)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            dsu (slot=1, port=1, init_addr=1)

            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=2){
        raw{
            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data
            dsu (slot=1, port=0, init_addr=2)
            rep (slot=1, port=0, level=0, iter=2, step=4, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=2, step=4, delay=0)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=2, step=4, delay=0)

            wait (cycle=0)
            act (mode=0, param=2, ports=0b1000)
            halt
        }
    }

    cell (x=1, y=2){
        raw{
            # build incoming route

            route (slot=0, option=0, sr=1, source=1, target=0b0110)
            route (slot=0, option=0, sr=0, source=3, target=128)
            act (mode=0, param=0, ports=0b0100)

            # connect RF1 to DPU4, RF2 to DPU5, DPU4 acc to RF3
            swb (slot=0, option=0, source=1, target=4)
            swb (slot=0, option=0, source=2, target=5)
            swb (slot=0, option=0, source=4, target=3)
            act (ports=1)

            # write to RF1
            dpu (slot=4, option=0, mode=1)
            act (ports=1, param=4)

            dsu (slot=1, port=2, init_addr=0)
            dsu (slot=2, port=2, init_addr=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu

            # read RF1 64 elements at a time, repeat 64 times, delay till RF2 reloaded 

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)



            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=16, step=1, delay=0)

            act (mode=0, param=1, ports=34)

            wait (cycle=0)
            act (mode=0, param=3, ports=0b0001)

            # read bulk RF3
            dsu (slot=3, port=3, init_addr=0)



            wait (cycle=13)
            act (ports=0b1000, param=3)
            halt
        }
    }

    cell (x=2, y=2){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=34)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            dsu (slot=1, port=1, init_addr=2)

            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=3){
        raw{
            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data
            dsu (slot=1, port=0, init_addr=3)
            rep (slot=1, port=0, level=0, iter=2, step=4, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=2, step=4, delay=0)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=2, step=4, delay=0)

            wait (cycle=0)
            act (mode=0, param=2, ports=0b1000)
            halt
        }
    }

    cell (x=1, y=3){
        raw{
            # build incoming route

            route (slot=0, option=0, sr=1, source=1, target=0b0110)
            route (slot=0, option=0, sr=0, source=3, target=128)
            act (mode=0, param=0, ports=0b0100)

            # connect RF1 to DPU4, RF2 to DPU5, DPU4 acc to RF3
            swb (slot=0, option=0, source=1, target=4)
            swb (slot=0, option=0, source=2, target=5)
            swb (slot=0, option=0, source=4, target=3)
            act (ports=1)

            # write to RF1
            dpu (slot=4, option=0, mode=1)
            act (ports=1, param=4)

            dsu (slot=1, port=2, init_addr=0)
            dsu (slot=2, port=2, init_addr=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu

            # read RF1 64 elements at a time, repeat 64 times, delay till RF2 reloaded 

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=16, step=1, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=16, step=1, delay=0)



            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=16, step=1, delay=0)

            act (mode=0, param=1, ports=34)

            wait (cycle=0)
            act (mode=0, param=3, ports=0b0001)

            # read bulk RF3
            dsu (slot=3, port=3, init_addr=0)



            wait (cycle=13)
            act (ports=0b1000, param=3)
            halt
        }
    }

    cell (x=2, y=3){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=34)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            dsu (slot=1, port=1, init_addr=3)

            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }


}
