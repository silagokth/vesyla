epoch <ep0_x> {
    cell (x=0, y=0){
        rop <route_c00_c10> (slot=0, port=2) {
            route (slot=0, sr=0, source=2, target=0b010000000)
        }

        rop <read_x_from_ib> (slot=1, port=0) {
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=2, step=1, delay=0)
        }

        rop <write_x_to_sram> (slot=1, port=2) {
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)
        }

        rop <read_x_from_sram> (slot=2, port=3) {
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=2, step=1, delay=0)
        }
    }

    cell (x=1, y=0) {
        rop <route_c10_c00> (slot=0, port=2) {
            route (slot=0, sr=1, source=1, target=0b110)
        }
        rop <write_x_to_rf1> (slot=1, port=2) {
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=2, step=1, delay=0)

        }
    }
}

epoch <ep1_w> {
    cell (x=0, y=0){
        rop <read_w_from_ib> (slot=1, port=0){
            dsu (slot=1, port=0, init_addr=2)
            rep (slot=1, port=0, level=0, iter=4, step=1, delay=0) 
            rep (slot=1, port=0, level=1, iter=8, step=4, delay=t1)
        }
        rop <write_w_to_sram> (slot=1, port=2){
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=4, step=1, delay=0)
            rep (slot=1, port=2, level=1, iter=8, step=0, delay=t1)
        }
        rop <read_w_from_sram> (slot=2, port=3){
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=4, step=1, delay=0)
            rep (slot=2, port=3, level=1, iter=8, step=0, delay=t1)
        }
    }
    cell (x=1, y=0){
        rop <route_c10_c20> (slot=0, port=2){
            route (slot=0, sr=1, source=1, target=0b110) ### need to set it again
            route (slot=0, sr=0, source=3, target=0b010000000)
        }
        rop <write_w_to_rf2> (slot=2, port=2){
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=4, step=1, delay=0)
            rep (slot=2, port=2, level=1, iter=8, step=0, delay=t1)
        }
        rop <swb> (slot=0, port=0) {
            swb (slot=0, option=0, channel=4, source=1, target=4)
            swb (slot=0, option=0, channel=5, source=2, target=5)
            swb (slot=0, option=0, channel=3, source=4, target=3)
        }
        rop <read_x_from_rf1> (slot=1, port=1){ 
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=32, step=1, delay=0) # Load input vector (x)
            rep (slot=1, port=1, level=1, iter=2, step=0, delay=0)  # Repeat 2 times 
            rep (slot=1, port=1, level=2, iter=8, step=0, delay=t2) # Repeat 8 times (2*8=16)
        }
        rop <read_w_from_rf2> (slot=2, port=1) {
            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=32, step=1, delay=0)
            rep (slot=2, port=1, level=1, iter=2, step=32, delay=0)
            rep (slot=2, port=1, level=2, iter=8, step=0, delay=t2)
        }
        rop <compute> (slot=4, port=0) {
            dpu (slot=4, mode=8)
            rep (slot=4, port=0, level=0, iter=2, step=0, delay=t3)#t3
            rep (slot=4, port=0, level=1, iter=8, step=0, delay=t3)#t4
        }
        rop <write_o_to_rf3> (slot=3, port=0) {
            dsu (slot=3, port=0, init_addr=0)
            rep (slot=3, port=0, level=0, iter=2, step=1, delay=t4)### t5 
            rep (slot=3, port=0, level=1, iter=8, step=2, delay=t4)### t6 
        }
        rop <read_o_from_rf3> (slot=3, port=3) {
            dsu (slot=3, port=3, init_addr=0)
        }
    }
    cell (x=2, y=0) {
        rop <route_c20_c10> (slot=0, port=2) {
            route (slot=0, sr=1, source=1, target=0b100)
        }
        rop <write_o_to_sram> (slot=2, port=2) {
            dsu (slot=2, port=2, init_addr=0)
        }
        rop <read_o_from_sram> (slot=1, port=3) {
            dsu (slot=1, port=3, init_addr=0)
        }
        rop <write_o_to_ob> (slot=1, port=1) {
            dsu (slot=1, port=1, init_addr=0)
        }
    }
}
