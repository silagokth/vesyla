epoch <rb0> {
        rop <route0r> (row=0, col=0, slot=0, port=2){
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
        }
        rop <input_r> (row=0, col=0, slot=1, port=0){
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=1, step=2, delay=0)
            rep (slot=1, port=0, level=1, iter=1, step=1, delay=0)
        }
        rop <input_w> (row=0, col=0, slot=1, port=2){
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, iter=3, step=1, delay=0)
        }
        cstr ( "input_r == input_w" )

        rop <read_ab> (row=0, col=0, slot=2, port=3){
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, iter=3, step=1, delay=0)
        }
        cstr ( "input_w < read_ab " )
        cstr ( "route0r < read_ab")

        rop <route1wr> (row=1, col=0, slot=0, port=2){
            route (slot=0, option=0, sr=1, source=1, target= 0b0000000000000110)
            route (slot=0, option=0, sr=0, source=3, target= 0b010000000)
        }
        rop <write_a> (row=1, col=0, slot=1, port=2){
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, iter=1, step=1, delay=t1)
        }

        rop <write_b> (row=1, col=0, slot=2, port=2){
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, iter=1, step=1, delay=t1)
        }
        cstr ( "route1wr < write_a ")
        cstr ( "route1wr < write_b ")
        cstr ( "read_ab.e0[0] == write_a.e0[0] ")
        cstr ( "read_ab.e0[1] == write_b.e0[0]" )
        cstr ( "read_ab.e0[2] == write_a.e0[1]" )
        cstr ( "read_ab.e0[3] == write_b.e0[1]")

        rop <swb> (row=1, col=0, slot=0, port=0){
            swb (slot=0, option=0, channel=4, source=1, target=4)
            swb (slot=0, option=0, channel=5, source=2, target=5)
            swb (slot=0, option=0, channel=3, source=4, target=3)
        }

        rop <read_a_seq> (row=1, col=0, slot=1, port=1){
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, iter=31, step=1, delay=0)
        }

        rop <read_b_seq> (row=1, col=0, slot=2, port=1){
            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, iter=31, step=1, delay=0)
        }
        cstr ( "write_a < read_a_seq ")
        cstr ( "write_b < read_b_seq" )
        cstr ( "swb < read_a_seq ")
        cstr ( "read_a_seq == read_b_seq ")

        rop <write_c_seq> (row=1, col=0, slot=3, port=0){
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, iter=31, step=1, delay=0)
        }
        cstr ( "write_c_seq == read_a_seq + 1 ")

        rop <compute> (row=1, col=0, slot=4, port=0){
            dpu (slot=4, mode=7)
        }
        cstr ( "read_a_seq + 1 > compute ")
        

        rop <read_c> (row=1, col=0, slot=3, port=3){
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, iter=1, step=1, delay=0)
        }
        cstr ( "read_c.e0[0] > write_c_seq.e0[15]" )
        cstr ( "read_c.e0[1] > write_c_seq.e0[31]" )


        rop <route2w> ( row=2, col=0, slot=0, port=2){
            route (slot=0, option=0, sr=1, source=1, target= 0b0000000000000100)
        }

        rop <write_c> ( row=2, col=0, slot=2, port=2){
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, iter=1, step=1, delay=0)
        }
        cstr ( "route2w < write_c" )
        cstr ( "write_c == read_c" )
        rop <output_r> (row=2, col=0, slot=1, port=3){
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, iter=1, step=1, delay=0)
        }

        rop <output_w> (row=2, col=0, slot=1, port=1){
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, iter=1, step=1, delay=0)
        }
        cstr ( "output_r > write_c" )
        cstr ( "output_r == output_w" )

        cstr("compute != swb")
        cstr("compute != route1wr")
        cstr("swb != route1wr")













    
 }