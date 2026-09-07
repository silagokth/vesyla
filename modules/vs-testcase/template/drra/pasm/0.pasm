epoch {
  # read the four bulks of the input buffer: a at addresses 0 and 1, b at 2
  # and 3. The inner rep steps by 2 so the two reads of one outer iteration
  # land on one a bulk and its matching b bulk.
  rop <input_r> (row=0, col=0, slot=1, port=0) {
    evt (init_addr=0)
    rep (iter=2, step=2, delay=0)
    rep (iter=2, step=1, delay=0)
  }

  rop <routes> (row=0, col=0, slot=0, port=1) {
    # route between slot 1 and center
    conf (variant="route", option=0, sr=0, source=1, target=0b10000)
    # route between center and slots 3 and 4
    conf (variant="route", option=0, sr=1, source=4, target=0b11000)

    # route between slot 2 and center
    conf (variant="route", option=1, sr=0, source=2, target=0b10000)
    # route between center and slot 1
    conf (variant="route", option=1, sr=1, source=4, target=0b10)

    # timing pattern for changing routes
    evt (port=1)
    rep (iter=2, step=1, delay=t2)
  }

  # write the a values to RF in slot 4 with some delay t1 between writes
  rop <write_a> (row=0, col=0, slot=4, port=2) {
    evt (init_addr=0)
    rep (iter=2, step=1, delay=t1)
  }

  # write the b values to RF in slot 3 with some delay t1 between writes
  rop <write_b> (row=0, col=0, slot=3, port=2) {
    evt (init_addr=0)
    rep (iter=2, step=1, delay=t1)
  }

  # configure swb to link RFs and DPU
  rop <swb> (row=0, col=0, slot=0, port=0) {
    conf (variant="swb", option=0, channel=5, source=4, target=5) # from slot 4 (first RF) to 5 (first input of DPU)
    conf (variant="swb", option=0, channel=6, source=3, target=6) # from slot 3 (second RF) to 6 (second input of DPU)
    conf (variant="swb", option=0, channel=2, source=5, target=2) # from slot 5 (output of DPU) to 2 (output RF)
    evt ()
  }

  # read the a values sequentially from the first RF (slot 4)
  rop <read_a_seq> (row=0, col=0, slot=4, port=1) {
    evt (init_addr=0)
    rep (iter=32, step=1, delay=0)
  }

  # read the b values sequentially from the second RF (slot 3)
  rop <read_b_seq> (row=0, col=0, slot=3, port=1) {
    evt (init_addr=0)
    rep (iter=32, step=1, delay=0)
  }

  # write the c values sequentially to the output RF (slot 2)
  rop <write_c_seq> (row=0, col=0, slot=2, port=0) {
    evt (init_addr=0)
    rep (iter=32, step=1, delay=0)
  }

  # configure the DPU to compute the product of the a and b values
  rop <compute> (row=0, col=0, slot=5, port=0) {
    conf (mode=7)
    evt ()
    rep (iter=32, step=0, delay=0)
  }

  # read the c values from the output RF (slot 2)
  rop <read_c> (row=0, col=0, slot=2, port=3) {
    evt (init_addr=0)
    rep (iter=2, step=1, delay=0)
  }

  # write the c values to the output buffer
  rop <output_w> (row=0, col=0, slot=1, port=1) {
    evt (init_addr=0)
    rep (iter=2, step=1, delay=0)
  }

  # List of constraints
  cstr ("input_r.[0].[0][0] == write_a.[0].[0]") # write a first part to RF after it is read from input buffer
  cstr ("input_r.[0].[0][1] == write_b.[0].[0]") # write b first part to RF after it is read from input buffer
  cstr ("input_r.[0].[1][0] == write_a.[0].[1]") # write a second part to RF after it is read from input buffer
  cstr ("input_r.[0].[1][1] == write_b.[0].[1]") # write b second part to RF after it is read from input buffer

  cstr ("write_a < read_a_seq") # start to read a sequentially from RF after writing it
  cstr ("write_b < read_b_seq") # start to read b sequentially from RF after writing it
  cstr ("swb < read_a_seq") # configure swb before reading the a values from RF
  cstr ("read_a_seq == read_b_seq") # read a and b values in parallel
  cstr ("read_a_seq == compute") # start reading a and b values at least one cycle before computing
  cstr ("write_c_seq == read_a_seq + 1") # start to write c values after reading a and b values
  cstr ("read_c.[0].[0] > write_c_seq.[0].[15]") # read c first part after the first 16 values are written to output RF
  cstr ("read_c.[0].[1] > write_c_seq.[0].[31]") # read c second part after the last 16 values are written to output RF

  cstr ("read_c == output_w") # write to output buffer synchronously when reading from output RF

  # constraints for route switching
  cstr ("routes.[0].[0] < write_a")
  cstr ("routes.[0].[0] < write_b")
  cstr ("routes.[0].[1] > write_b.[0].[1]")
  cstr ("routes.[0].[1] < output_w")
}
