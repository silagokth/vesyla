module {
  "pasm.make_epoch" () <{id="xxx"}> ({
    %10 = "pasm.make_rop"() <{id = "xxx", row=1:i32, col=1:i32, slot=1:i32, port=1:i32}> ({
      "pasm.make_instr"() <{id = "xxx", type="dpu", param={col=1:i32, row=2:i32, port=2:i32, slot=3:i32}}> : () -> ()
      "pasm.yield"() : () -> ()
    }) : () -> !pasm.rop
    %11 = "pasm.make_anchor" (%10) < {event = 0 : i32, slice = [2, 3]} > : (!pasm.rop) -> !pasm.anchor
    "pasm.make_binary_constraint" (%11, %11) < {type = "linear", param = {comparator= ">"}} > : (!pasm.anchor, !pasm.anchor) -> ()
    "pasm.yield"() : () -> ()
  }) : () -> ()

  "pasm.make_epoch" () <{id="yyy"}> ({
    %10 = "pasm.make_cop"() <{id = "xxx", row=1:i32, col=1:i32}> ({
      "pasm.make_instr"() <{id = "xxx", type="wait", param={mode = 0: i32, param = 5: i32}}> : () -> ()
      "pasm.make_instr"() <{id = "xxx", type="halt", param={}}> : () -> ()
      "pasm.yield"() : () -> ()
    }) : () -> !pasm.cop
    "pasm.yield"() : () -> ()
  }) : () -> ()
}
