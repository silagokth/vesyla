epoch {
    rop <route0r> (row=0, col=0, slot=0, port=2){
        route (option=0, sr=0, source=2, target= 0b010000000)
    }
    rop <input_r> (row=0, col=0, slot=1, port=0){
        dsu (init_addr=0)
        rep (level=0, iter=1, step=2, delay=0)
        rep (level=1, iter=1, step=1, delay=0)
    }
    rop <input_w> (row=0, col=0, slot=1, port=2){
        dsu (init_addr=0)
        rep (iter=3, step=1, delay=0)
    }
    rop <read_ab> (row=0, col=0, slot=2, port=3){
        dsu (init_addr=0)
        rep (iter=3, step=1, delay=0)
    }
    rop <route1wr> (row=1, col=0, slot=0, port=2){
        route (option=0, sr=1, source=1, target= 0b0000000000000110)
        route (option=0, sr=0, source=3, target= 0b010000000)
    }
    rop <write_a> (row=1, col=0, slot=1, port=2){
        dsu (init_addr=0)
        rep (iter=1, step=1, delay=t1)
    }
    rop <write_b> (row=1, col=0, slot=2, port=2){
        dsu (init_addr=0)
        rep (iter=1, step=1, delay=t1)
    }
    rop <swb> (row=1, col=0, slot=0, port=0){
        swb (option=0, channel=4, source=1, target=4)
        swb (option=0, channel=5, source=2, target=5)
        swb (option=0, channel=3, source=4, target=3)
    }
    rop <read_a_seq> (row=1, col=0, slot=1, port=1){
        dsu (init_addr=0)
        rep (iter=31, step=1, delay=0)
    }
    rop <read_b_seq> (row=1, col=0, slot=2, port=1){
        dsu (init_addr=0)
        rep (iter=31, step=1, delay=0)
    }
    rop <write_c_seq> (row=1, col=0, slot=3, port=0){
        dsu (init_addr=0)
        rep (iter=31, step=1, delay=0)
    }
    rop <compute> (row=1, col=0, slot=4, port=0){
        dpu (mode=7)
    }
    rop <read_c> (row=1, col=0, slot=3, port=3){
        dsu (init_addr=0)
        rep (iter=1, step=1, delay=0)
    }
    rop <route2w> (row=2, col=0, slot=0, port=2){
        route (option=0, sr=1, source=1, target= 0b0000000000000100)
    }

    rop <write_c> (row=2, col=0, slot=2, port=2){
        dsu (init_addr=0)
        rep (iter=1, step=1, delay=0)
    }

    rop <output_r> (row=2, col=0, slot=1, port=3){
        dsu (init_addr=0)
        rep (iter=1, step=1, delay=0)
    }

    rop <output_w> (row=2, col=0, slot=1, port=1){
        dsu (init_addr=0)
        rep (iter=1, step=1, delay=0)
    }

	cstr (" input_r == input_w ")
	cstr (" input_w < read_ab ")
	cstr (" route0r < read_ab ")
	cstr (" route1wr < write_a ")
	cstr (" route1wr < write_b ")
	cstr (" read_ab.e0[0] == write_a.e0[0] ")
	cstr (" read_ab.e0[1] == write_b.e0[0] ")
	cstr (" read_ab.e0[2] == write_a.e0[1] ")
	cstr (" read_ab.e0[3] == write_b.e0[1] ")
	cstr (" write_a < read_a_seq ")
	cstr (" write_b < read_b_seq ")
	cstr (" swb < read_a_seq ")
	cstr (" read_a_seq == read_b_seq ")
	cstr (" read_a_seq + 1 > compute ")
	cstr (" write_c_seq == read_a_seq + 1 ")

	cstr (" read_c.e0[0] > write_c_seq.e0[15] ")
	cstr (" read_c.e0[1] > write_c_seq.e0[31] ")

	cstr (" write_c == read_c ")
	cstr (" output_r > write_c ")
	cstr (" output_r == output_w ")


}
