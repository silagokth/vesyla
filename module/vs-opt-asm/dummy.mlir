module {
  %0 = "cidfg.IntegerConstant"() <{value = 10 : i32}> : () -> !cidfg.Integer<32, "myCustomValue">
  %1 = "cidfg.IntegerConstant"() <{value = 1 : i32}> : () -> !cidfg.Integer<32, "myCustomValue">
  %2 = "cidfg.IntegerConstant"() <{value = 2 : i32}> : () -> !cidfg.Integer<32, "myCustomValue">
  %8 = "cidfg.scalar_binop"(%0, %1) <{op = "add"}> : (!cidfg.Integer<32, "myCustomValue">, !cidfg.Integer<32, "myCustomValue">) -> !cidfg.Integer<32, "myCustomValue">
  %3 = "cidfg.AffineIndex"(%0, %8, %2) : (!cidfg.Integer<32, "myCustomValue">, !cidfg.Integer<32, "myCustomValue">, !cidfg.Integer<32, "myCustomValue">) -> !cidfg.AffineIndex<1>
  %4 = "cidfg.allocate"() <{bind = "reg", chunk_num = 10 : i32, chunk_size = 10 : i32}> : () -> !cidfg.Array<10, 10, "reg">
  %5 = "cidfg.read"(%4, %3) : (!cidfg.Array<10, 10, "reg">, !cidfg.AffineIndex<1>) -> !cidfg.Stream<10, 10>
  %6 = "cidfg.unary_compute" (%5) <{op = "tanh"}> : (!cidfg.Stream<10, 10>) -> !cidfg.Stream<10, 10>
  %7 = "cidfg.write"(%4, %6, %3) : (!cidfg.Array<10, 10, "reg">, !cidfg.Stream<10, 10>, !cidfg.AffineIndex<1>) -> !cidfg.Array<10, 10, "reg">

}