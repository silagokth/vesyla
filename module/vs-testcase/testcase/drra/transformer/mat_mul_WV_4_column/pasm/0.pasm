epoch <rb0> {
    cell (x=0, y=0){
        raw{
            # load data
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)
            act (mode=0, param=2, ports=0b1000)


            # load w_matrix into IOSRAM
            dsu (slot=1, port=0, init_addr=1)
            rep (slot=1, port=0, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=0, level=0, iter=0, step=0, delay=0)
            # write w_matrix to IOSRAM
            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=2, level=0, iter=0, step=0, delay=0)


            #read IOSRAM
            dsu (slot=2, port=3, init_addr=32)
            rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=4, step=4, delay=60)
            repx (slot=2, port=3, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=3, level=2, iter=1, step=0, delay=60)
            repx (slot=2, port=3, level=2, iter=0, step=0, delay=0)

            wait (cycle=18)
            act (mode=0, param=1, ports=0b0101)
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
            wait (cycle=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)
            act (mode=0, param=1, ports=0b0100)

            # write to RF2
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
            repx (slot=2, port=2, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=2, level=1, iter=4, step=0, delay=60)
            repx (slot=2, port=2, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=2, level=2, iter=1, step=0, delay=60)
            repx (slot=2, port=2, level=2, iter=0, step=0, delay=0)

            wait (cycle=5)


            # read from RF1 RF2, write to dpu

            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
            rep (slot=1, port=1, level=1, iter=0, step=0, delay=0)
            repx (slot=1, port=1, level=1, iter=1, step=0, delay=0)
            rep (slot=1, port=1, level=2, iter=1, step=4, delay=0)
            repx (slot=1, port=1, level=2, iter=0, step=0, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=0, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=1, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=4, step=0, delay=0)
            repx (slot=2, port=1, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=2, iter=1, step=0, delay=0)
            repx (slot=2, port=1, level=2, iter=0, step=0, delay=0)




            dpu (slot=4, option=0, mode=8)
            rep (slot=4, level=0, iter=0, step=0, delay=3)
            repx (slot=4, level=0, iter=1, step=0, delay=0)
            rep (slot=4, level=1, iter=1, step=0, delay=3)
            repx (slot=4, level=1, iter=0, step=0, delay=0)


            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            act (mode=0, param=2, ports=0b0100)

            act (mode=0, param=1, ports=34)
            act (ports=1, param=4)

            rep (slot=3, port=0, level=0, iter=0, step=1, delay=3)
            repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
            rep (slot=3, port=0, level=1, iter=1, step=0, delay=3)
            act (ports=1, param=3)




            # read bulk RF3, wait till next 16, repeat 16 times
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=4, step=1, delay=63)
            rep (slot=3, port=3, level=1, iter=1, step=0, delay=63)

            wait (cycle=56)
            act (ports=0b1000, param=3)
            halt
        }

    }

    cell (x=2, y=0){
        raw{
           # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=106)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=63)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=0, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)



            wait (cycle=182)
            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=1){
        raw{
            # load data
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)
            act (mode=0, param=2, ports=0b1000)


            # load w_matrix into IOSRAM
            dsu (slot=1, port=0, init_addr=1)
            rep (slot=1, port=0, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=0, level=0, iter=0, step=0, delay=0)
            # write w_matrix to IOSRAM
            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=2, level=0, iter=0, step=0, delay=0)


            #read IOSRAM
            dsu (slot=2, port=3, init_addr=32)
            rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=4, step=4, delay=60)
            repx (slot=2, port=3, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=3, level=2, iter=1, step=0, delay=60)
            repx (slot=2, port=3, level=2, iter=0, step=0, delay=0)

            wait (cycle=18)
            act (mode=0, param=1, ports=0b0101)
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
        wait (cycle=0)
        dsu (slot=1, port=2, init_addr=0)
        rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)
        act (mode=0, param=1, ports=0b0100)

        # write to RF2
        dsu (slot=2, port=2, init_addr=0)
        rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
        repx (slot=2, port=2, level=0, iter=0, step=0, delay=0)
        rep (slot=2, port=2, level=1, iter=4, step=0, delay=60)
        repx (slot=2, port=2, level=1, iter=0, step=0, delay=0)
        rep (slot=2, port=2, level=2, iter=1, step=0, delay=60)
        repx (slot=2, port=2, level=2, iter=0, step=0, delay=0)

        wait (cycle=5)


        # read from RF1 RF2, write to dpu

        dsu (slot=1, port=1, init_addr=4)
        rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
        rep (slot=1, port=1, level=1, iter=0, step=0, delay=0)
        repx (slot=1, port=1, level=1, iter=1, step=0, delay=0)
        rep (slot=1, port=1, level=2, iter=1, step=4, delay=0)
        repx (slot=1, port=1, level=2, iter=0, step=0, delay=0)

        dsu (slot=2, port=1, init_addr=0)
        rep (slot=2, port=1, level=0, iter=0, step=1, delay=0)
        repx (slot=2, port=1, level=0, iter=1, step=0, delay=0)
        rep (slot=2, port=1, level=1, iter=4, step=0, delay=0)
        repx (slot=2, port=1, level=1, iter=0, step=0, delay=0)
        rep (slot=2, port=1, level=2, iter=1, step=0, delay=0)
        repx (slot=2, port=1, level=2, iter=0, step=0, delay=0)


        dpu (slot=4, option=0, mode=8)
        rep (slot=4, level=0, iter=0, step=0, delay=3)
        repx (slot=4, level=0, iter=1, step=0, delay=0)
        rep (slot=4, level=1, iter=1, step=0, delay=3)
        repx (slot=4, level=1, iter=0, step=0, delay=0)


        # write to RF3 from dpu
        dsu (slot=3, port=0, init_addr=0)
        act (mode=0, param=2, ports=0b0100)

        act (mode=0, param=1, ports=34)
        act (ports=1, param=4)

        rep (slot=3, port=0, level=0, iter=0, step=1, delay=3)
        repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
        rep (slot=3, port=0, level=1, iter=1, step=0, delay=3)
        act (ports=1, param=3)




        # read bulk RF3, wait till next 16, repeat 16 times
        dsu (slot=3, port=3, init_addr=0)
        rep (slot=3, port=3, level=0, iter=4, step=1, delay=63)
        rep (slot=3, port=3, level=1, iter=1, step=0, delay=63)

        wait (cycle=56)
        act (ports=0b1000, param=3)
            halt
        }
    }

    cell (x=2, y=1){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=106)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=63)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=0, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=4)
            rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)



            wait (cycle=182)
            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }




    cell (x=0, y=2){
        raw{
            # load data
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)
            act (mode=0, param=2, ports=0b1000)


            # load w_matrix into IOSRAM
            dsu (slot=1, port=0, init_addr=1)
            rep (slot=1, port=0, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=0, level=0, iter=0, step=0, delay=0)
            # write w_matrix to IOSRAM
            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=2, level=0, iter=0, step=0, delay=0)


            #read IOSRAM
            dsu (slot=2, port=3, init_addr=32)
            rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=4, step=4, delay=60)
            repx (slot=2, port=3, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=3, level=2, iter=1, step=0, delay=60)
            repx (slot=2, port=3, level=2, iter=0, step=0, delay=0)

            wait (cycle=18)
            act (mode=0, param=1, ports=0b0101)
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
            wait (cycle=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)
            act (mode=0, param=1, ports=0b0100)

            # write to RF2
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
            repx (slot=2, port=2, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=2, level=1, iter=4, step=0, delay=60)
            repx (slot=2, port=2, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=2, level=2, iter=1, step=0, delay=60)
            repx (slot=2, port=2, level=2, iter=0, step=0, delay=0)

            wait (cycle=5)


            # read from RF1 RF2, write to dpu
            dsu (slot=1, port=1, init_addr=8)
            rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
            rep (slot=1, port=1, level=1, iter=0, step=0, delay=0)
            repx (slot=1, port=1, level=1, iter=1, step=0, delay=0)
            rep (slot=1, port=1, level=2, iter=1, step=4, delay=0)
            repx (slot=1, port=1, level=2, iter=0, step=0, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=0, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=1, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=4, step=0, delay=0)
            repx (slot=2, port=1, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=2, iter=1, step=0, delay=0)
            repx (slot=2, port=1, level=2, iter=0, step=0, delay=0)




            dpu (slot=4, option=0, mode=8)
            rep (slot=4, level=0, iter=0, step=0, delay=3)
            repx (slot=4, level=0, iter=1, step=0, delay=0)
            rep (slot=4, level=1, iter=1, step=0, delay=3)
            repx (slot=4, level=1, iter=0, step=0, delay=0)


            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            act (mode=0, param=2, ports=0b0100)

            act (mode=0, param=1, ports=34)
            act (ports=1, param=4)

            rep (slot=3, port=0, level=0, iter=0, step=1, delay=3)
            repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
            rep (slot=3, port=0, level=1, iter=1, step=0, delay=3)
            act (ports=1, param=3)




            # read bulk RF3, wait till next 16, repeat 16 times
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=4, step=1, delay=63)
            rep (slot=3, port=3, level=1, iter=1, step=0, delay=63)

            wait (cycle=56)
            act (ports=0b1000, param=3)
            halt
        }
    }

    cell (x=2, y=2){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=106)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=63)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=0, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=8)
            rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)



            wait (cycle=182)
            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }

    cell (x=0, y=3){
        raw{
            # load data
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)

            # build route
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
            act (mode=0, param=0, ports=0b0100)

            # read x_matrix first
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=1, step=1, delay=0)
            act (mode=0, param=1, ports=0b0101)
            act (mode=0, param=2, ports=0b1000)


            # load w_matrix into IOSRAM
            dsu (slot=1, port=0, init_addr=1)
            rep (slot=1, port=0, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=0, level=0, iter=0, step=0, delay=0)
            # write w_matrix to IOSRAM
            dsu (slot=1, port=2, init_addr=32)
            rep (slot=1, port=2, level=0, iter=16, step=1, delay=0)
            repx (slot=1, port=2, level=0, iter=0, step=0, delay=0)


            #read IOSRAM
            dsu (slot=2, port=3, init_addr=32)
            rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=4, step=4, delay=60)
            repx (slot=2, port=3, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=3, level=2, iter=1, step=0, delay=60)
            repx (slot=2, port=3, level=2, iter=0, step=0, delay=0)

            wait (cycle=18)
            act (mode=0, param=1, ports=0b0101)
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
            wait (cycle=0)
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=1, step=1, delay=0)
            act (mode=0, param=1, ports=0b0100)

            # write to RF2
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
            repx (slot=2, port=2, level=0, iter=0, step=0, delay=0)
            rep (slot=2, port=2, level=1, iter=4, step=0, delay=60)
            repx (slot=2, port=2, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=2, level=2, iter=1, step=0, delay=60)
            repx (slot=2, port=2, level=2, iter=0, step=0, delay=0)

            wait (cycle=5)


            # read from RF1 RF2, write to dpu
            dsu (slot=1, port=1, init_addr=12)
            rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
            rep (slot=1, port=1, level=1, iter=0, step=0, delay=0)
            repx (slot=1, port=1, level=1, iter=1, step=0, delay=0)
            rep (slot=1, port=1, level=2, iter=1, step=4, delay=0)
            repx (slot=1, port=1, level=2, iter=0, step=0, delay=0)

            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=0, step=1, delay=0)
            repx (slot=2, port=1, level=0, iter=1, step=0, delay=0)
            rep (slot=2, port=1, level=1, iter=4, step=0, delay=0)
            repx (slot=2, port=1, level=1, iter=0, step=0, delay=0)
            rep (slot=2, port=1, level=2, iter=1, step=0, delay=0)
            repx (slot=2, port=1, level=2, iter=0, step=0, delay=0)




            dpu (slot=4, option=0, mode=8)
            rep (slot=4, level=0, iter=0, step=0, delay=3)
            repx (slot=4, level=0, iter=1, step=0, delay=0)
            rep (slot=4, level=1, iter=1, step=0, delay=3)
            repx (slot=4, level=1, iter=0, step=0, delay=0)


            # write to RF3 from dpu
            dsu (slot=3, port=0, init_addr=0)
            act (mode=0, param=2, ports=0b0100)

            act (mode=0, param=1, ports=34)
            act (ports=1, param=4)

            rep (slot=3, port=0, level=0, iter=0, step=1, delay=3)
            repx (slot=3, port=0, level=0, iter=1, step=0, delay=0)
            rep (slot=3, port=0, level=1, iter=1, step=0, delay=3)
            act (ports=1, param=3)




            # read bulk RF3, wait till next 16, repeat 16 times
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=4, step=1, delay=63)
            rep (slot=3, port=3, level=1, iter=1, step=0, delay=63)

            wait (cycle=56)
            act (ports=0b1000, param=3)

            halt
        }
    }

    cell (x=2, y=3){
        raw{
            # build route
            route (slot=0, option=0, sr=1, source=1, target= 0b0100)
            act (mode=0, param=0, ports=0b0100)

            wait (cycle=106)


            # write data 
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=63)
            act (mode=0, param=2, ports=0b0100)

            # store data
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=3, level=0, iter=0, step=0, delay=0)
            dsu (slot=1, port=1, init_addr=12)
            rep (slot=1, port=1, level=0, iter=4, step=1, delay=0)
            repx (slot=1, port=1, level=0, iter=0, step=0, delay=0)



            wait (cycle=182)
            act (mode=0, param=1, ports=0b1010)
            halt
        }
    }


}
