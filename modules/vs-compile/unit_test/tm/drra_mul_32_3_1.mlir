module {
  pasm.epoch< {id = "DdF4HHOR"}> {
    // CELL [0,0] — Input scratchpad
    // Routing configuration: send from slot2 toward south (intercell bulk out)
    pasm.rop< {col = 0 : i32, id = "route0r", port = 1 : i32, row = 0 : i32, slot = 0 : i32}> {
      pasm.instr< {id = "G768j3gj", param = {option = 0 : i32, source = 2 : i32, sr = 0 : i32, target = 128 : i32}, type = "route"}>
      pasm.yield
    }

    // rop <input_r>: Read from input buffer
    // Two-level affine access: addresses 0,2,1,3 (interleaved A and B rows)
    // Outer loop: step=2, iter=2 → starts at 0, 1
    // Inner loop: step=2, iter=2 → jumps by 2 within each outer iteration
    // Together they produce address sequence: 0, 2, 1, 3
    affine.for %outer = 0 to 2 {
      affine.for %inner = 0 to 2 {
        %addr = affine.apply affine_map<(d1,d0) -> (d1*2 + d0 + 0)> (%outer, %inner)
        pasm.rop< {col = 0 : i32, id = "input_r", port = 0 : i32, row = 0 : i32, slot = 1 : i32}> {
          pasm.instr< {id = "gvgyzgRJ", param = {init_addr = 0 : i32}, type = "dsu"}>
        }
      }
    }

    // rop <read_ab>: Read from iosram, send bulk toward south to cell[1,0]
    // Sequential read of all 4 rows (A[0-15], B[0-15], A[16-31], B[16-31])
    affine.for %i = 0 to 4 {
      pasm.rop< {col = 0 : i32, id = "input_w", port = 2 : i32, row = 0 : i32, slot = 1 : i32}> {
        pasm.instr< {id = "pw4qnaNy", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
      pasm.rop< {col = 0 : i32, id = "read_ab", port = 3 : i32, row = 0 : i32, slot = 2 : i32}> {
        pasm.instr< {id = "W62iqJ5a", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
    }

    // CELL [1,0] — Compute cell

    // Routing configuration:
    //   receive from north into slot1 and slot2 (A and B)
    //   send from slot3 toward south (C)
    pasm.rop< {col = 0 : i32, id = "route1wr", port = 1 : i32, row = 1 : i32, slot = 0 : i32}> {
      pasm.instr< {id = "LRrNp49z", param = {option = 0 : i32, source = 1 : i32, sr = 1 : i32, target = 6 : i32}, type = "route"}>
      pasm.instr< {id = "Mjxza4D0", param = {option = 0 : i32, source = 3 : i32, sr = 0 : i32, target = 128 : i32}, type = "route"}>
      pasm.yield
    }

    // rop <write_a> and <write_b>: Receive bulk from north, write to RF
    // Each bulk row contains 16 elements, 2 rows per vector
    // The delay=t1 between iterations is left symbolic for the scheduler
    affine.for %i = 0 to 2 {
      // Write A chunk (rows 0 and 2 of iosram → RF A addresses 0-15 and 16-31)
      pasm.rop< {col = 0 : i32, id = "write_a", port = 2 : i32, row = 1 : i32, slot = 1 : i32}> {
        pasm.instr< {id = "M3YfCR5f", param = {init_addr = 0 : i32}, type = "dsu"}>
      }

      // Write B chunk (rows 1 and 3 of iosram → RF B addresses 0-15 and 16-31)
      pasm.rop< {col = 0 : i32, id = "write_b", port = 2 : i32, row = 1 : i32, slot = 2 : i32}> {
        pasm.instr< {id = "IsXOlUBN", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
    }

    // Switchbox configuration: word-level paths
    //   RF1(slot1) → DPU input channel 4
    //   RF2(slot2) → DPU input channel 5
    //   DPU output channel 3 → RF3(slot3)
    pasm.rop< {col = 0 : i32, id = "swb", port = 0 : i32, row = 1 : i32, slot = 0 : i32}> {
      pasm.instr< {id = "YdA5u3Yy", param = {channel = 4 : i32, option = 0 : i32, source = 1 : i32, target = 4 : i32}, type = "swb"}>
      pasm.instr< {id = "OSzWi0gG", param = {channel = 5 : i32, option = 0 : i32, source = 2 : i32, target = 5 : i32}, type = "swb"}>
      pasm.instr< {id = "r1SILSTC", param = {channel = 3 : i32, option = 0 : i32, source = 4 : i32, target = 3 : i32}, type = "swb"}>
      pasm.yield
    }

    // rop <read_a_seq>, <read_b_seq>, <write_c_seq>:
    // Word-level sequential read of A and B, compute multiply, write C
    // These three rops are tightly coupled:
    //   read_a_seq == read_b_seq (same cycle)
    //   write_c_seq == read_a_seq + 1 (1 cycle DPU latency)
    affine.for %i = 0 to 32 {
      pasm.rop< {col = 0 : i32, id = "read_a_seq", port = 1 : i32, row = 1 : i32, slot = 1 : i32}> {
        pasm.instr< {id = "sRTtRYOJ", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
      pasm.rop< {col = 0 : i32, id = "read_b_seq", port = 1 : i32, row = 1 : i32, slot = 2 : i32}> {
        pasm.instr< {id = "XLLP9sTg", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
      pasm.rop< {col = 0 : i32, id = "write_c_seq", port = 0 : i32, row = 1 : i32, slot = 3 : i32}> {
        pasm.instr< {id = "pFnab1jT", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
    }

    pasm.rop< {col = 0 : i32, id = "compute", port = 0 : i32, row = 1 : i32, slot = 4 : i32}> {
      pasm.instr< {id = "iL6Z3Q4a", param = {mode = 7 : i32}, type = "dpu"}>
      pasm.yield
    }

    // rop <read_c>: Read C from RF3 in bulk, send toward south
    // 2 bulk iterations (32 elements / 16 per bulk row)
    affine.for %i = 0 to 2 {
      pasm.rop< {col = 0 : i32, id = "read_c", port = 3 : i32, row = 1 : i32, slot = 3 : i32}> {
        pasm.instr< {id = "yeJoY5gh", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
    }

    // CELL [2,0] — Output scratchpad

    // Routing configuration: receive from north into slot2
    pasm.rop< {col = 0 : i32, id = "route2w", port = 1 : i32, row = 2 : i32, slot = 0 : i32}> {
      pasm.instr< {id = "ChLHwLyK", param = {option = 0 : i32, source = 1 : i32, sr = 1 : i32, target = 4 : i32}, type = "route"}>
      pasm.yield
    }

    // rop <write_c>: Receive bulk from north, write to iosram_out
    affine.for %i = 0 to 2 {
      pasm.rop< {col = 0 : i32, id = "write_c", port = 2 : i32, row = 2 : i32, slot = 2 : i32}> {
        pasm.instr< {id = "MviN9anx", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
      pasm.rop< {col = 0 : i32, id = "output_r", port = 3 : i32, row = 2 : i32, slot = 1 : i32}> {
        pasm.instr< {id = "Xu9DPoET", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
      pasm.rop< {col = 0 : i32, id = "output_w", port = 1 : i32, row = 2 : i32, slot = 1 : i32}> {
        pasm.instr< {id = "Cm57Ahs5", param = {init_addr = 0 : i32}, type = "dsu"}>
      }
    }

    // Timing constraints
    pasm.cstr< {expr = "input_r == input_w", type = "linear"}>
    pasm.cstr< {expr = "route0r < read_ab", type = "linear"}>
    pasm.cstr< {expr = "route1wr < write_a", type = "linear"}>
    pasm.cstr< {expr = "route1wr < write_b", type = "linear"}>
    pasm.cstr< {expr = "read_ab > input_w", type = "linear"}>
    pasm.cstr< {expr = "read_ab.e0[0] == write_a.e0[0]", type = "linear"}>
    pasm.cstr< {expr = "read_ab.e0[1] == write_b.e0[0]", type = "linear"}>
    pasm.cstr< {expr = "read_ab.e0[2] == write_a.e0[1]", type = "linear"}>
    pasm.cstr< {expr = "read_ab.e0[3] == write_b.e0[1]", type = "linear"}>
    pasm.cstr< {expr = "write_a < read_a_seq", type = "linear"}>
    pasm.cstr< {expr = "write_b < read_b_seq", type = "linear"}>
    pasm.cstr< {expr = "swb < read_a_seq", type = "linear"}>
    pasm.cstr< {expr = "read_a_seq == read_b_seq", type = "linear"}>
    pasm.cstr< {expr = "read_a_seq == compute", type = "linear"}>
    pasm.cstr< {expr = "write_c_seq == read_a_seq + 1", type = "linear"}>
    pasm.cstr< {expr = "read_c.e0[0] > write_c_seq.e0[15]", type = "linear"}>
    pasm.cstr< {expr = "read_c.e0[1] > write_c_seq.e0[31]", type = "linear"}>
    pasm.cstr< {expr = "write_c == read_c", type = "linear"}>
    pasm.cstr< {expr = "output_r > write_c", type = "linear"}>
    pasm.cstr< {expr = "output_r == output_w", type = "linear"}>
  }
}
