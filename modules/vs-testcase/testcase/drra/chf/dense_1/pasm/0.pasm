epoch <ep1> { # Read IB to RF1
    cell (x=0, y=0) {
        rop <route0r> (slot=0, port=2) {
            route (slot=0, sr=0, source=2, target=0b010000000)
        }

        rop <input_from_ib> (slot=1, port=0) { # Read input from input buffer
            dsu (slot=1, port=0, init_addr=0)
            rep (slot=1, port=0, level=0, iter=1, step=1, delay=0)
        }

        rop <input_to_sram> (slot=1, port=2) { # Write input to SRAM
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, iter=1, step=1, delay=0)
        }

        rop <read_input_from_sram> (slot=2, port=3) { # Read input from SRAM
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, iter=1, step=1, delay=0)
        }
    }
    cell (x=1, y=0) {
        rop <route1w> (slot=0, port=2) {
            route (slot=0, sr=1, source=1, target=0b110) # Route input to RF1 + RF2
        }

        rop <write_input_to_rf1> (slot=1, port=2) { # Write input to RF1
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, iter=1, step=1, delay=0)
        }
    }
}

epoch <ep2> {
    cell (x=0, y=0) {
        rop <weight_from_ib> (slot=1, port=0) { # Read weight matrix from input buffer
            dsu (slot=1, port=0, init_addr=2)
            rep (slot=1, port=0, level=0, iter=3, step=1, delay=0) # Load 4 rows
            rep (slot=1, port=0, level=1, iter=31, step=4, delay=t1) # Load 32 times (4x32=128)
            rep (slot=1, port=0, level=2, iter=1, step=128, delay=t2) # Load 2 times (128x2=256)
        }

        rop <weight_to_sram> (slot=1, port=2) { # Write weight matrix to SRAM
            dsu (slot=1, port=2, init_addr=0)
            rep (slot=1, port=2, level=0, iter=3, step=1, delay=0) # Load 4 rows
            rep (slot=1, port=2, level=1, iter=31, step=0, delay=t1) # Load 32 times (4x32=128)
            rep (slot=1, port=2, level=2, iter=1, step=0, delay=t2) # Load 2 times (128x2=256)
        }

        rop <read_weight_from_sram> (slot=2, port=3) { # Read weight matrix from SRAM
            dsu (slot=2, port=3, init_addr=0)
            rep (slot=2, port=3, level=0, iter=3, step=1, delay=0) # Load 4 rows
            rep (slot=2, port=3, level=1, iter=31, step=0, delay=t1) # Load 32 times (4x32=128)
            rep (slot=2, port=3, level=2, iter=1, step=0, delay=t2) # Load 2 times (128x2=256)
        }
    }

    cell (x=1, y=0) {
        rop <route2w> (slot=0, port=2) {
            route (slot=0, sr=1, source=1, target=0b110) # Route input to RF1 + RF2
            route (slot=0, sr=0, source=3, target=0b010000000) # Route input to RF3
        }

        rop <swb> (slot=0, port=0) {
            swb (slot=0, channel=4, source=1, target=4)
            swb (slot=0, channel=5, source=2, target=5)
            swb (slot=0, channel=3, source=4, target=3)
        }

        rop <write_weight_to_rf2> (slot=2, port=2) {
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=0) # Load 4 rows
            rep (slot=2, port=2, level=1, iter=31, step=0, delay=t1) # Load 32 times (4x32=128)
            rep (slot=2, port=2, level=2, iter=1, step=0, delay=t2) # Load 2 times (128x2=256)
        }

        rop <read_rf1_to_dpu> (slot=1, port=1) {
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=23, step=1, delay=0) # Load 24 values
            rep (slot=1, port=1, level=1, iter=1, step=0, delay=0) # Repeat 2 times (24x2=48)
            rep (slot=1, port=1, level=2, iter=31, step=0, delay=t3) # Load 32 times (2x32=64)
            rep (slot=1, port=1, level=3, iter=1, step=0, delay=t4) # Load 2 times (64x2=128)
        }

        rop <read_rf2_to_dpu> (slot=2, port=1) {
            dsu (slot=2, port=1, init_addr=0)
            rep (slot=2, port=1, level=0, iter=23, step=1, delay=0) # Load 24 values
            rep (slot=2, port=1, level=1, iter=1, step=32, delay=0) # Repeat 2 times (24x2=48) (for next rows)
            rep (slot=2, port=1, level=2, iter=31, step=0, delay=t3) # Load 32 times (2x32=64)
            rep (slot=2, port=1, level=3, iter=1, step=0, delay=t4) # Load 2 times (64x2=128)
        }

        rop <compute> (slot=4, port=0) {
            dpu (slot=4, mode=8)
            rep (slot=4, port=0, level=0, iter=1, step=0, delay=t8) # Write 1 number
            rep (slot=4, port=0, level=1, iter=31, step=0, delay=t5) # Repeat 32 times (1x32=32)
            rep (slot=4, port=0, level=2, iter=1, step=0, delay=t6) # Repeat 2 times (32x2=64)
        }

        rop <write_output_to_rf3> (slot=3, port=0) {
            dsu (slot=3, port=0, init_addr=0) # Write output to RF3 (1 word)
            rep (slot=3, port=0, level=0, iter=1, step=1, delay=t8) # Repeat 2 times (1x2=2 words)
            rep (slot=3, port=0, level=1, iter=31, step=2, delay=t5) # Repeat 32 times (2x32=64 words)
            rep (slot=3, port=0, level=2, iter=1, step=0, delay=t6) # Repeat 2 times (2x64=128 words)
        }

        rop <read_rf3_to_sram> (slot=3, port=3) {
            dsu (slot=3, port=3, init_addr=0) # Read output from RF3 (bulk read of 16 words)
            rep (slot=3, port=3, level=0, iter=3, step=1, delay=0) # Repeat 4 times (4x16=64 words)
            rep (slot=3, port=3, level=1, iter=1, step=0, delay=t7) # Repeat 2 times (64x2=128 words)
        }
    }

    cell (x=2, y=0) {
        rop <route3w> (slot=0, port=2) {
            route (slot=0, sr=1, source=1, target=0b100) # Route RF3 to IOSRAM2
        }

        rop <write_rf3_to_sram> (slot=2, port=2) {
            dsu (slot=2, port=2, init_addr=0)
            rep (slot=2, port=2, level=0, iter=3, step=1, delay=0) # Load 4 rows
            rep (slot=2, port=2, level=1, iter=1, step=4, delay=t7) # Repeat 2 times (4x2=8rows)
        }

        rop <read_sram_to_ob> (slot=1, port=3) {
            dsu (slot=1, port=3, init_addr=0)
            rep (slot=1, port=3, level=0, iter=7, step=1, delay=0) # Load 8 rows
        }

        rop <output_to_ob> (slot=1, port=1) {
            dsu (slot=1, port=1, init_addr=0)
            rep (slot=1, port=1, level=0, iter=7, step=1, delay=0) # Load 8 rows
        }
    }
}