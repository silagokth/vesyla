epoch <rb0> {
    cell (x=0, y=0){
        rop <input_r_x> (slot=1, port=0) {
            dsu (slot=1, port=0, init_addr=256)
            rep (slot=1, port=0, level=0, iter=3, step=1, delay=0)
        }
        rop <input_w_x> (slot=1, port=2) {
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
        }
        rop <route0r> (slot=0, port=2) {
            route (slot=0, option=0, sr=0, source=2, target= 0b010000000)
        }
        rop <read_x> (slot=2, port=3) {
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
        }
    }

    cell (x=1, y=0){
        rop <route1wr> (slot=0, port=2) {
            route (slot=0, option=0, sr=1, source=1, target= 0b0000000000000110)
            route (slot=0, option=0, sr=0, source=3, target= 0b010000000)
        }
        rop <write_x> (slot=1, port=2) {
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
        }
    }
}

epoch <rb1> {
    cell (x=0, y=0){
        rop <input_r_b> (slot=1, port=0) {
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=3, step=1, delay=0)
            rep (slot=1, port=0, level=1, iter=63, step=4, delay=t1)
        }
        rop <input_w_b> (slot=1, port=2) {
            dsu (slot=1, port=2, init_addr=2)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=0)
            rep (slot=1, port=2, level=1, iter=63, step=0, delay=t1)
        }
        rop <read_b> (slot=2, port=3) {
            dsu (slot=2, port=3, init_addr=2)
            rep (slot=2, port=3, level=0, iter=3, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=63, step=0, delay=t1)
        }
    }
    cell (x=1, y=0){
        rop <route1wr> (slot=0, port=2) {
            route (slot=0, option=0, sr=1, source=1, target= 0b0000000000000110)
            route (slot=0, option=0, sr=0, source=3, target= 0b010000000)
        }
        rop <write_b> (slot=2, port=2) {
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
            rep (slot=2, port=2, level=1, iter=63, step=0, delay=t1)
        }
        rop <swb> (slot=0, port=0) {
            swb (slot=0, option=0, channel=4, source=1, target=4)
            swb (slot=0, option=0, channel=5, source=2, target=5)
            swb (slot=0, option=0, channel=3, source=4, target=3)
        }
        rop <read_a_seq> (slot=1, port=1) {
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=63, step=1, delay=0)
            rep (slot=1, port=1, level=1, iter=63, step=0, delay=t2)
        }
        rop <read_b_seq> (slot=2, port=1) {
            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=63, step=1, delay=0)
            rep (slot=2, port=1, level=1, iter=63, step=0, delay=t2)
        }
        rop <write_c_seq> (slot=3, port=0) {
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=63, step=1, delay=t4)
        }
        rop <compute> (slot=4, port=0) {
            dpu (slot=4, mode=8)
            rep (slot=4, level=0, iter=63, step=1, delay=t4)
        }
        rop <read_c> (slot=3, port=3) {
            dsu (slot=3, port=3, init_addr=0)
            rep (slot=3, port=3, level=0, iter=3, step=1, delay=0)
        }
    }
    cell (x=2, y=0){
        rop <route2w> (slot=0, port=2) {
            route (slot=0, option=0, sr=1, source=1, target= 0b0000000000000100)
        }
        rop <write_c> (slot=2, port=2) {
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=0)
        }
        rop <output_r> (slot=1, port=3) {
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=3, step=1, delay=0)
        }
        rop <output_w> (slot=1, port=1) {
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=3, step=1, delay=0)
        }
    }
}
