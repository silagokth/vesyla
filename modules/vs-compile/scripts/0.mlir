module {
  pasm.epoch< {id = "xnNre0qB"}> {
    pasm.rop< {col = 0 : i32, port = 0 : i32, row = 0 : i32, slot = 1 : i32, sym_name = "input_r"}> {
      pasm.instr< {id = "KIvvlVlh", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 2 : i32, row = 0 : i32, slot = 1 : i32, sym_name = "input_w"}> {
      pasm.instr< {id = "rqwZKkHQ", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.cstr< {dst = @input_w, max_delay = 0 : i32, min_delay = 0 : i32, src = @input_r}>
    pasm.rop< {col = 0 : i32, port = 1 : i32, row = 0 : i32, slot = 0 : i32, sym_name = "route_io_swb"}> {
      pasm.instr< {id = "uTAEIctU", param = {option = 0 : i32, source = 2 : i32, sr = 0 : i32, target = 128 : i32}, type = "route"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 3 : i32, row = 0 : i32, slot = 2 : i32, sym_name = "read_vec"}> {
      pasm.instr< {id = "GzhqFqdd", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.cstr< {dst = @read_vec, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @input_w}>
    pasm.cstr< {dst = @read_vec, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @route_io_swb}>
    pasm.rop< {col = 0 : i32, port = 1 : i32, row = 1 : i32, slot = 0 : i32, sym_name = "route_swb_rf1"}> {
      pasm.instr< {id = "CMNYfy5E", param = {option = 0 : i32, source = 1 : i32, sr = 1 : i32, target = 2 : i32}, type = "route"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 2 : i32, row = 1 : i32, slot = 1 : i32, sym_name = "write_vec"}> {
      pasm.instr< {id = "rrdpRk5H", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.cstr< {dst = @write_vec, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @route_swb_rf1}>
    pasm.cstr< {dst = @write_vec, max_delay = 0 : i32, min_delay = 0 : i32, src = @read_vec}>
    pasm.rop< {col = 0 : i32, port = 0 : i32, row = 1 : i32, slot = 0 : i32, sym_name = "swb"}> {
      pasm.instr< {id = "t3J8W2S4", param = {channel = 4 : i32, option = 0 : i32, source = 1 : i32, target = 4 : i32}, type = "swb"}>
      pasm.instr< {id = "k0kteBMR", param = {channel = 5 : i32, option = 0 : i32, source = 2 : i32, target = 5 : i32}, type = "swb"}>
      pasm.instr< {id = "lZpsOkur", param = {channel = 2 : i32, option = 0 : i32, source = 4 : i32, target = 2 : i32}, type = "swb"}>
      pasm.instr< {id = "sOWfwQlm", param = {channel = 4 : i32, option = 1 : i32, source = 1 : i32, target = 4 : i32}, type = "swb"}>
      pasm.instr< {id = "Dukdl0XR", param = {channel = 5 : i32, option = 1 : i32, source = 2 : i32, target = 5 : i32}, type = "swb"}>
      pasm.instr< {id = "O68J7IaJ", param = {channel = 3 : i32, option = 1 : i32, source = 4 : i32, target = 3 : i32}, type = "swb"}>
      pasm.instr< {id = "SDzslkZ1", param = {port = 0 : i32}, type = "evt"}>
      pasm.instr< {id = "ItVg96IH", param = {delay = "t2", iter = 2 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 1 : i32, row = 1 : i32, slot = 1 : i32, sym_name = "read_a"}> {
      pasm.instr< {id = "aqkPf6eJ", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr< {id = "WcQxlo6H", param = {delay = "t0", iter = 8 : i32, step = 1 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 1 : i32, row = 1 : i32, slot = 2 : i32, sym_name = "read_b"}> {
      pasm.instr< {id = "LuxRSMIE", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr< {id = "zbi3Xp8C", param = {delay = "t0", iter = 8 : i32, step = 0 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.cstr< {dst = @read_b, max_delay = 0 : i32, min_delay = 0 : i32, src = @read_a}>
    pasm.cstr< {dst = @read_a, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @write_vec}>
    pasm.cstr< {dst = @read_a, dst_event = "e0", dst_idx_hi = array<i32: 0>, dst_idx_lo = array<i32: 0>, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @swb, src_event = "e0", src_idx_hi = array<i32: 0>, src_idx_lo = array<i32: 0>}>
    pasm.cstr< {dst = @swb, dst_event = "e0", dst_idx_hi = array<i32: 1>, dst_idx_lo = array<i32: 1>, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @write_acc_rf2, src_event = "e0", src_idx_hi = array<i32: 6>, src_idx_lo = array<i32: 6>}>
    pasm.rop< {col = 0 : i32, port = 0 : i32, row = 1 : i32, slot = 2 : i32, sym_name = "write_acc_rf2"}> {
      pasm.instr< {id = "PiR0cw8U", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.instr< {id = "qOIv1C0g", param = {delay = "t1", iter = 7 : i32, step = 0 : i32}, type = "rep"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 0 : i32, row = 1 : i32, slot = 3 : i32, sym_name = "write_acc_rf3"}> {
      pasm.instr< {id = "wnxEz4Gt", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 0 : i32, row = 1 : i32, slot = 4 : i32, sym_name = "compute_start"}> {
      pasm.instr< {id = "VpktUitn", param = {mode = 1 : i32}, type = "dpu"}>
      pasm.yield
    }
    pasm.cstr< {dst = @compute_start, max_delay = 0 : i32, min_delay = 0 : i32, src = @read_a}>
    pasm.cstr< {dst = @write_acc_rf2, dst_event = "e0", dst_idx_hi = array<i32: 0>, dst_idx_lo = array<i32: 0>, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 0>, src_idx_lo = array<i32: 0>}>
    pasm.cstr< {dst = @read_a, dst_event = "e0", dst_idx_hi = array<i32: 1>, dst_idx_lo = array<i32: 1>, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @write_acc_rf2, src_event = "e0", src_idx_hi = array<i32: 0>, src_idx_lo = array<i32: 0>}>
    pasm.cstr< {dst = @write_acc_rf2, dst_event = "e0", dst_idx_hi = array<i32: 1>, dst_idx_lo = array<i32: 1>, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 1>, src_idx_lo = array<i32: 1>}>
    pasm.cstr< {dst = @write_acc_rf2, dst_event = "e0", dst_idx_hi = array<i32: 2>, dst_idx_lo = array<i32: 2>, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 2>, src_idx_lo = array<i32: 2>}>
    pasm.cstr< {dst = @write_acc_rf2, dst_event = "e0", dst_idx_hi = array<i32: 3>, dst_idx_lo = array<i32: 3>, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 3>, src_idx_lo = array<i32: 3>}>
    pasm.cstr< {dst = @write_acc_rf2, dst_event = "e0", dst_idx_hi = array<i32: 4>, dst_idx_lo = array<i32: 4>, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 4>, src_idx_lo = array<i32: 4>}>
    pasm.cstr< {dst = @write_acc_rf2, dst_event = "e0", dst_idx_hi = array<i32: 5>, dst_idx_lo = array<i32: 5>, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 5>, src_idx_lo = array<i32: 5>}>
    pasm.cstr< {dst = @write_acc_rf2, dst_event = "e0", dst_idx_hi = array<i32: 6>, dst_idx_lo = array<i32: 6>, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 6>, src_idx_lo = array<i32: 6>}>
    pasm.cstr< {dst = @write_acc_rf3, max_delay = 1 : i32, min_delay = 1 : i32, src = @read_a, src_event = "e0", src_idx_hi = array<i32: 7>, src_idx_lo = array<i32: 7>}>
    pasm.rop< {col = 0 : i32, port = 1 : i32, row = 1 : i32, slot = 0 : i32, sym_name = "route_rf_swb"}> {
      pasm.instr< {id = "o8chsa15", param = {option = 0 : i32, source = 3 : i32, sr = 0 : i32, target = 128 : i32}, type = "route"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 3 : i32, row = 1 : i32, slot = 3 : i32, sym_name = "read_c"}> {
      pasm.instr< {id = "k7qbAf8u", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.cstr< {dst = @read_c, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @route_rf_swb}>
    pasm.cstr< {dst = @read_c, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @write_acc_rf3}>
    pasm.rop< {col = 0 : i32, port = 1 : i32, row = 2 : i32, slot = 0 : i32, sym_name = "route_swb_sram"}> {
      pasm.instr< {id = "uvx6p3pB", param = {option = 0 : i32, source = 1 : i32, sr = 1 : i32, target = 4 : i32}, type = "route"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 2 : i32, row = 2 : i32, slot = 2 : i32, sym_name = "write_c"}> {
      pasm.instr< {id = "GNsyvbcz", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.cstr< {dst = @write_c, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @route_swb_sram}>
    pasm.cstr< {dst = @read_c, max_delay = 0 : i32, min_delay = 0 : i32, src = @write_c}>
    pasm.rop< {col = 0 : i32, port = 3 : i32, row = 2 : i32, slot = 1 : i32, sym_name = "output_r"}> {
      pasm.instr< {id = "Z2WGSMB2", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.rop< {col = 0 : i32, port = 1 : i32, row = 2 : i32, slot = 1 : i32, sym_name = "output_w"}> {
      pasm.instr< {id = "XpUHKS2R", param = {init_addr = 0 : i32}, type = "dsu"}>
      pasm.yield
    }
    pasm.cstr< {dst = @output_r, max_delay = 10000000 : i32, min_delay = 1 : i32, src = @write_c}>
    pasm.cstr< {dst = @output_w, max_delay = 0 : i32, min_delay = 0 : i32, src = @output_r}>
    pasm.yield
  }
}
