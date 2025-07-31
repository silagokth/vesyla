module {
  pasm.epoch<{id = "kTpEZj4q"}> {
    pasm.rop<{col = 0 : i32, id = "route0r", port = 2 : i32, row = 0 : i32, slot = 0 : i32}> {
      pasm.instr<{id = "mJwEwY3q", param = {option = 0 : i32, source = 2 : i32, sr = 0 : i32, target = 128 : i32}, type = "route"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "input_r", port = 0 : i32, row = 0 : i32, slot = 1 : i32}> {
      pasm.instr<{id = "RI2C00Ui", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "F5qNc3MJ", param = {delay = 0 : i32, iter = 1 : i32, level = 0 : i32, step = 2 : i32}, type = "rep"}>
      pasm.instr<{id = "j8HJkNPi", param = {delay = 0 : i32, iter = 1 : i32, level = 1 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "input_w", port = 2 : i32, row = 0 : i32, slot = 1 : i32}> {
      pasm.instr<{id = "tAmJYdxH", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "BpHcdc8Z", param = {delay = 0 : i32, iter = 3 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "read_ab", port = 3 : i32, row = 0 : i32, slot = 2 : i32}> {
      pasm.instr<{id = "Nmmygx5j", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "tAsTN6zi", param = {delay = 0 : i32, iter = 3 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "route1wr", port = 2 : i32, row = 1 : i32, slot = 0 : i32}> {
      pasm.instr<{id = "KZpU2cZc", param = {option = 0 : i32, source = 1 : i32, sr = 1 : i32, target = 6 : i32}, type = "route"}>
      pasm.instr<{id = "vf4j508Q", param = {option = 0 : i32, source = 3 : i32, sr = 0 : i32, target = 128 : i32}, type = "route"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "write_a", port = 2 : i32, row = 1 : i32, slot = 1 : i32}> {
      pasm.instr<{id = "tiC7T7gB", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "RNUJJIpD", param = {delay = "t1", iter = 1 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "write_b", port = 2 : i32, row = 1 : i32, slot = 2 : i32}> {
      pasm.instr<{id = "vVVYvs01", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "XuYIiUZ9", param = {delay = "t1", iter = 1 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "swb", port = 0 : i32, row = 1 : i32, slot = 0 : i32}> {
      pasm.instr<{id = "SZ6IVbJQ", param = {channel = 4 : i32, option = 0 : i32, source = 1 : i32, target = 4 : i32}, type = "swb"}>
      pasm.instr<{id = "GbZvjDyF", param = {channel = 5 : i32, option = 0 : i32, source = 2 : i32, target = 5 : i32}, type = "swb"}>
      pasm.instr<{id = "sJdH1T6F", param = {channel = 3 : i32, option = 0 : i32, source = 4 : i32, target = 3 : i32}, type = "swb"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "read_a_seq", port = 1 : i32, row = 1 : i32, slot = 1 : i32}> {
      pasm.instr<{id = "vSLjmijb", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "IdjT4sji", param = {delay = 0 : i32, iter = 31 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "read_b_seq", port = 1 : i32, row = 1 : i32, slot = 2 : i32}> {
      pasm.instr<{id = "t7TrAFwW", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "sPdEiYHh", param = {delay = 0 : i32, iter = 31 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "write_c_seq", port = 0 : i32, row = 1 : i32, slot = 3 : i32}> {
      pasm.instr<{id = "KSET1nsx", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "pRE99ngG", param = {delay = 0 : i32, iter = 31 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "compute", port = 0 : i32, row = 1 : i32, slot = 4 : i32}> {
      pasm.instr<{id = "Kxxi3j2F", param = {mode = 7 : i32}, type = "dpu"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "read_c", port = 3 : i32, row = 1 : i32, slot = 3 : i32}> {
      pasm.instr<{id = "kUJUsQzY", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "x4pWfXJk", param = {delay = 0 : i32, iter = 1 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "route2w", port = 2 : i32, row = 2 : i32, slot = 0 : i32}> {
      pasm.instr<{id = "UNilyEsX", param = {option = 0 : i32, source = 1 : i32, sr = 1 : i32, target = 4 : i32}, type = "route"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "write_c", port = 2 : i32, row = 2 : i32, slot = 2 : i32}> {
      pasm.instr<{id = "rd3qDwuz", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "Y3IwI7Kn", param = {delay = 0 : i32, iter = 1 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "output_r", port = 3 : i32, row = 2 : i32, slot = 1 : i32}> {
      pasm.instr<{id = "P09VNG50", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "LdbFfHaV", param = {delay = 0 : i32, iter = 1 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop<{col = 0 : i32, id = "output_w", port = 1 : i32, row = 2 : i32, slot = 1 : i32}> {
      pasm.instr<{id = "dU9oEtbI", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr<{id = "Rh4tfCWU", param = {delay = 0 : i32, iter = 1 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.cstr<{expr = "input_r == input_w", type = "linear"}>
    pasm.cstr<{expr = "route0r <read_ab", type = "linear"}>
    pasm.cstr<{expr = "route1wr <write_a", type = "linear"}>
    pasm.cstr<{expr = "route1wr <write_b", type = "linear"}>
    pasm.cstr<{expr = "read_ab > input_w", type = "linear"}>
    pasm.cstr<{expr = "read_ab.e0[0] == write_a.e0[0]", type = "linear"}>
    pasm.cstr<{expr = "read_ab.e0[1] == write_b.e0[0]", type = "linear"}>
    pasm.cstr<{expr = "read_ab.e0[2] == write_a.e0[1]", type = "linear"}>
    pasm.cstr<{expr = "read_ab.e0[3] == write_b.e0[1]", type = "linear"}>
    pasm.cstr<{expr = "write_a <read_a_seq", type = "linear"}>
    pasm.cstr<{expr = "write_b <read_b_seq", type = "linear"}>
    pasm.cstr<{expr = "swb <read_a_seq", type = "linear"}>
    pasm.cstr<{expr = "read_a_seq == read_b_seq", type = "linear"}>
    pasm.cstr<{expr = "read_a_seq + 1 > compute", type = "linear"}>
    pasm.cstr<{expr = "write_c_seq == read_a_seq + 1", type = "linear"}>
    pasm.cstr<{expr = "read_c.e0[0] > write_c_seq.e0[15]", type = "linear"}>
    pasm.cstr<{expr = "read_c.e0[1] > write_c_seq.e0[31]", type = "linear"}>
    pasm.cstr<{expr = "write_c == read_c", type = "linear"}>
    pasm.cstr<{expr = "output_r > write_c", type = "linear"}>
    pasm.cstr<{expr = "output_r == output_w", type = "linear"}>
    pasm.yield
  }
}
