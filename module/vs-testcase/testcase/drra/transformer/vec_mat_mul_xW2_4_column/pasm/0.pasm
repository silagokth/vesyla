epoch <rb0> {
    cell (x=0, y=0){
        raw{
            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data from buffer for x vector
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)


            # load data from buffer for w matrix
            dsu (slot=1, port=0, init_addr=16)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            # 1024/16 = 64 iter=63
            rep (slot=1, port=0, level=1, iter=15, step=16, delay=48)
            repx (slot=1, port=0, level=1, iter=-1, step=0, delay=3)

            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            # 1024/16 = 64 iter=63
            rep (slot=1, port=2, level=1, iter=15, step=0, delay=48)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=3)
            wait (cycle=6)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix and w_matrix in alternation
            #wait (cycle=12)
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=32, delay=0)
            rep (slot=2, port=3, level=1, iter=3, step=1, delay=0)
            #repeat 4 times
            rep (slot=2, port=3, level=2, iter=3, step=4, delay=56)
            repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)
            #repeat 64 times
            rep (slot=2, port=3, level=3, iter=15, step=0, delay=56)
            repx (slot=2, port=3, level=3, iter=-1, step=0, delay=0)

            wait (cycle=1)
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


            # read RF1 64 elements at a time, repeat 64 times
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)
            # repeat 256 times
            rep (slot=1, port=1, level=2, iter=15, step=0, delay=0)
            repx (slot=1, port=1, level=2, iter=-1, step=0, delay=0)

            # read RF2 64 elements at a time, repeat 64 times

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)
            # repeat 256 times
            rep (slot=2, port=1, level=2, iter=15, step=0, delay=0)


            # write to RF1 and RF2
            wait (cycle=2)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=1, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=1, port=2, level=2, iter=-1, step=0, delay=0)
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=2, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=2, port=2, level=2, iter=-1, step=0, delay=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu


            repx (slot=2, port=1, level=2, iter=-1, step=0, delay=0)

            #dpu repeat 64 times
            dpu (slot=4, option=0, mode=8)
            rep (slot=4, level=0, iter=15, step=0, delay=63)
            repx (slot=4, level=0, iter=-1, step=0, delay=3)

            act (mode=0, param=1, ports=34)
            act (ports=1, param=4)

            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=15, step=1, delay=63)
            repx (slot=3, port=0, level=0, iter=-1, step=0, delay=3)
            wait (cycle=251)
            act (ports=1, param=3)

            # read bulk RF3, wait till next 16, repeat 16 times
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=3, port=3, level=0, iter=-1, step=0, delay=0)

            wait (cycle=3839)
            act (ports=0b1000, param=3)

            halt
        }

    }

    cell (x=2, y=0){
        raw{
          # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            # store to IO
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=-1, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=-1, step=0, delay=0)


            wait (cycle=4129)
            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)
            repx (slot=2, port=2, level=0, iter=-1, step=0, delay=0)
            act (mode=0, param=2, ports=0b0100)


            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=1){
        raw{
            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data from buffer for x vector
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)


            # load data from buffer for w matrix
            dsu (slot=1, port=0, init_addr=272)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            # 1024/16 = 64 iter=63
            rep (slot=1, port=0, level=1, iter=15, step=16, delay=48)
            repx (slot=1, port=0, level=1, iter=-1, step=0, delay=3)

            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            # 1024/16 = 64 iter=63
            rep (slot=1, port=2, level=1, iter=15, step=0, delay=48)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=3)
            wait (cycle=6)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix and w_matrix in alternation
            #wait (cycle=11)
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=32, delay=0)
            rep (slot=2, port=3, level=1, iter=3, step=1, delay=0)
            #repeat 4 times
            rep (slot=2, port=3, level=2, iter=3, step=4, delay=56)
            repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)
            #repeat 64 times
            rep (slot=2, port=3, level=3, iter=15, step=0, delay=56)
            repx (slot=2, port=3, level=3, iter=-1, step=0, delay=0)

            wait (cycle=1)
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


            # read RF1 64 elements at a time, repeat 64 times

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)
            # repeat 256 times
            rep (slot=1, port=1, level=2, iter=15, step=0, delay=0)
            repx (slot=1, port=1, level=2, iter=-1, step=0, delay=0)
        
            # read RF2 64 elements at a time, repeat 64 times

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)
            # repeat 256 times
            rep (slot=2, port=1, level=2, iter=15, step=0, delay=0)


            # write to RF1 and RF2
            wait (cycle=2)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=1, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=1, port=2, level=2, iter=-1, step=0, delay=0)
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=2, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=2, port=2, level=2, iter=-1, step=0, delay=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu


            repx (slot=2, port=1, level=2, iter=-1, step=0, delay=0)

            #dpu repeat 64 times
            dpu (slot=4, option=0, mode=8)
            rep (slot=4, level=0, iter=15, step=0, delay=63)
            repx (slot=4, level=0, iter=-1, step=0, delay=3)

            act (mode=0, param=1, ports=34)
            act (ports=1, param=4)

            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=15, step=1, delay=63)
            repx (slot=3, port=0, level=0, iter=-1, step=0, delay=3)
            wait (cycle=251)
            act (ports=1, param=3)

            # read bulk RF3, wait till next 16, repeat 16 times
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=3, port=3, level=0, iter=-1, step=0, delay=0)

            wait (cycle=3839)
            act (ports=0b1000, param=3)
            halt
        }
    }

    cell (x=2, y=1){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            # store to IO
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=-1, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=1)
            rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=-1, step=0, delay=0)


            wait (cycle=4129)
            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)
            repx (slot=2, port=2, level=0, iter=-1, step=0, delay=0)
            act (mode=0, param=2, ports=0b0100)


            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=2){
        raw{
            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data from buffer for x vector
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)


            # load data from buffer for w matrix
            dsu (slot=1, port=0, init_addr=528)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            rep (slot=1, port=0, level=1, iter=15, step=16, delay=48)
            repx (slot=1, port=0, level=1, iter=-1, step=0, delay=3)

            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            rep (slot=1, port=2, level=1, iter=15, step=0, delay=48)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=3)
            wait (cycle=6)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix and w_matrix in alternation
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=32, delay=0)
            rep (slot=2, port=3, level=1, iter=3, step=1, delay=0)
            rep (slot=2, port=3, level=2, iter=3, step=4, delay=56)
            repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)
            rep (slot=2, port=3, level=3, iter=15, step=0, delay=56)
            repx (slot=2, port=3, level=3, iter=-1, step=0, delay=0)

            wait (cycle=1)
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


            # read RF1 64 elements at a time, repeat 64 times, delay till RF2 reloaded 

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)
            rep (slot=1, port=1, level=2, iter=15, step=0, delay=0)
            repx (slot=1, port=1, level=2, iter=-1, step=0, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)
            rep (slot=2, port=1, level=2, iter=15, step=0, delay=0)


            # write to RF1 and RF2
            wait (cycle=2)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=1, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=1, port=2, level=2, iter=-1, step=0, delay=0)
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=2, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=2, port=2, level=2, iter=-1, step=0, delay=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu


            repx (slot=2, port=1, level=2, iter=-1, step=0, delay=0)

            dpu (slot=4, option=0, mode=8)
            rep (slot=4, level=0, iter=15, step=0, delay=63)
            repx (slot=4, level=0, iter=-1, step=0, delay=3)

            act (mode=0, param=1, ports=34)
            act (ports=1, param=4)

            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=15, step=1, delay=63)
            repx (slot=3, port=0, level=0, iter=-1, step=0, delay=3)
            wait (cycle=251)
            act (ports=1, param=3)

            # read bulk RF3, wait till next 16, repeat 16 times
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=3, port=3, level=0, iter=-1, step=0, delay=0)

            wait (cycle=3839)
            act (ports=0b1000, param=3)

            halt
        }
    }

    cell (x=2, y=2){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            # store to IO
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=-1, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=2)
            rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=-1, step=0, delay=0)


            wait (cycle=4129)
            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)
            repx (slot=2, port=2, level=0, iter=-1, step=0, delay=0)
            act (mode=0, param=2, ports=0b0100)


            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=3){
        raw{
            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # load data from buffer for x vector
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)


            # load data from buffer for w matrix
            dsu (slot=1, port=0, init_addr=784)
            rep (slot=1, port=0, level=0, iter=15, step=1, delay=0)
            rep (slot=1, port=0, level=1, iter=15, step=16, delay=48)
            repx (slot=1, port=0, level=1, iter=-1, step=0, delay=3)

            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=15, step=1, delay=0)
            rep (slot=1, port=2, level=1, iter=15, step=0, delay=48)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=3)
            wait (cycle=6)
            act (mode=0, param=1, ports=0b0101)

            # read x_matrix and w_matrix in alternation
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=32, delay=0)
            rep (slot=2, port=3, level=1, iter=3, step=1, delay=0)
            rep (slot=2, port=3, level=2, iter=3, step=4, delay=56)
            repx (slot=2, port=3, level=2, iter=-1, step=0, delay=0)
            rep (slot=2, port=3, level=3, iter=15, step=0, delay=56)
            repx (slot=2, port=3, level=3, iter=-1, step=0, delay=0)

            wait (cycle=1)
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


            # read RF1 64 elements at a time, repeat 64 times, delay till RF2 reloaded 

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=1, port=1, level=1, iter=3, step=0, delay=0)
            rep (slot=1, port=1, level=2, iter=15, step=0, delay=0)
            repx (slot=1, port=1, level=2, iter=-1, step=0, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=-1, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=3, step=0, delay=0)
            rep (slot=2, port=1, level=2, iter=15, step=0, delay=0)


            # write to RF1 and RF2
            wait (cycle=2)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=1, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=1, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=1, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=1, port=2, level=2, iter=-1, step=0, delay=0)
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=1)
            rep (slot=2, port=2, level=1, iter=3, step=0, delay=57)
            repx (slot=2, port=2, level=1, iter=-1, step=0, delay=0)
            rep (slot=2, port=2, level=2, iter=15, step=0, delay=57)
            repx (slot=2, port=2, level=2, iter=-1, step=0, delay=0)

            act (mode=0, param=1, ports=0b0100)
            act (mode=0, param=2, ports=0b0100)

            # read from RF1 RF2, write to dpu


            repx (slot=2, port=1, level=2, iter=-1, step=0, delay=0)

            dpu (slot=4, option=0, mode=8)
            rep (slot=4, level=0, iter=15, step=0, delay=63)
            repx (slot=4, level=0, iter=-1, step=0, delay=3)

            act (mode=0, param=1, ports=34)
            act (ports=1, param=4)

            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=15, step=1, delay=63)
            repx (slot=3, port=0, level=0, iter=-1, step=0, delay=3)
            wait (cycle=251)
            act (ports=1, param=3)

            # read bulk RF3, wait till next 16, repeat 16 times
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=3, port=3, level=0, iter=-1, step=0, delay=0)

            wait (cycle=3839)
            act (ports=0b1000, param=3)
            halt
        }
    }

    cell (x=2, y=3){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            # store to IO
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=-1, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=3)
            rep (slot=1, port=1, level=0, iter=0, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=-1, step=0, delay=0)


            wait (cycle=4129)
            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=0, step=1, delay=0)
            repx (slot=2, port=2, level=0, iter=-1, step=0, delay=0)
            act (mode=0, param=2, ports=0b0100)


            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }


}
