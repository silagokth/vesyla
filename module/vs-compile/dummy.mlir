module {
  "pasm.epoch" () <{id="epoch0"}> ({
    "pasm.rop"() <{id = "read_a", row=1:i32, col=0:i32, slot=1:i32, port=1:i32}> ({
      "pasm.instr"() <{id = "instr0", type="dsu", param={col=1:i32, row=2:i32, port=2:i32, slot=3:i32}}> : () -> ()
      "pasm.instr"() <{id = "instr1", type="rep", param={col=2:i32, row=3:i32, port=3:i32, slot=4:i32, level=0:i32, iter=3:i32, delay="t0"}}> : () -> ()
      "pasm.instr"() <{id = "instr2", type="rep", param={col=3:i32, row=4:i32, port=4:i32, slot=5:i32, level=1:i32, iter=5:i32, delay="t1"}}> : () -> ()
      "pasm.yield"() : () -> ()
    }) : () -> ()
    "pasm.rop"() <{id = "read_b", row=0:i32, col=0:i32, slot=1:i32, port=0:i32}> ({
      "pasm.instr"() <{id = "instr3", type="dsu", param={col=1:i32, row=2:i32, port=2:i32, slot=3:i32}}> : () -> ()
      "pasm.instr"() <{id = "instr4", type="rep", param={col=2:i32, row=3:i32, port=3:i32, slot=4:i32, level=0:i32, iter=3:i32, delay="t2"}}> : () -> ()
      "pasm.instr"() <{id = "instr5", type="rep", param={col=3:i32, row=4:i32, port=4:i32, slot=5:i32, level=1:i32, iter=5:i32, delay="t3"}}> : () -> ()
      "pasm.yield"() : () -> ()
    }) : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a == read_b"} > : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a.e0[0] == read_b.e0[0]"} > : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a.e0[1] == read_b.e0[1]"} > : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a.e0[1][2] == read_b.e0[1][2]"} > : () -> ()
    "pasm.yield"() : () -> ()
  }) : () -> ()

  "pasm.epoch" () <{id="epoch1"}> ({
    "pasm.rop"() <{id = "read_a", row=1:i32, col=0:i32, slot=1:i32, port=1:i32}> ({
      "pasm.instr"() <{id = "instr0", type="dsu", param={col=1:i32, row=2:i32, port=2:i32, slot=3:i32}}> : () -> ()
      "pasm.instr"() <{id = "instr1", type="rep", param={col=2:i32, row=3:i32, port=3:i32, slot=4:i32, level=0:i32, iter=3:i32, delay="t0"}}> : () -> ()
      "pasm.instr"() <{id = "instr2", type="rep", param={col=3:i32, row=4:i32, port=4:i32, slot=5:i32, level=1:i32, iter=5:i32, delay="t1"}}> : () -> ()
      "pasm.yield"() : () -> ()
    }) : () -> ()
    "pasm.rop"() <{id = "read_b", row=0:i32, col=0:i32, slot=1:i32, port=0:i32}> ({
      "pasm.instr"() <{id = "instr3", type="dsu", param={col=1:i32, row=2:i32, port=2:i32, slot=3:i32}}> : () -> ()
      "pasm.instr"() <{id = "instr4", type="rep", param={col=2:i32, row=3:i32, port=3:i32, slot=4:i32, level=0:i32, iter=3:i32, delay="t2"}}> : () -> ()
      "pasm.instr"() <{id = "instr5", type="rep", param={col=3:i32, row=4:i32, port=4:i32, slot=5:i32, level=1:i32, iter=5:i32, delay="t3"}}> : () -> ()
      "pasm.yield"() : () -> ()
    }) : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a == read_b"} > : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a.e0[0] == read_b.e0[0]"} > : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a.e0[1] == read_b.e0[1]"} > : () -> ()
    "pasm.constraint" () < {type = "linear", expr = "read_a.e0[1][2] == read_b.e0[1][2]"} > : () -> ()
    "pasm.yield"() : () -> ()
  }) : () -> ()
}
