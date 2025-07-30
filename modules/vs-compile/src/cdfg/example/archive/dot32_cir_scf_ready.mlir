!s32i = !cir.int<s, 32>
#fn_attr = #cir<extra({inline = #cir.inline<no>, nothrow = #cir.nothrow, optnone = #cir.optnone, uwtable = #cir.uwtable<async>})>
module @"/home/paul/Develop/vesyla/modules/vs-compile/src/cdfg/example/dot32.cpp" attributes {cir.lang = #cir.lang<cxx>, cir.sob = #cir.signed_overflow_behavior<undefined>, cir.triple = "x86_64-unknown-linux-gnu", cir.type_size_info = #cir.type_size_info<char = 8, int = 32, size_t = 64>, cir.uwtable = #cir.uwtable<async>, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e">} {
  cir.func dso_local @_Z5dot32PKfS0_i(%arg0: !cir.ptr<!cir.float>, %arg1: !cir.ptr<!cir.float>, %arg2: !s32i) -> !cir.float extra(#fn_attr) {
    %0 = cir.alloca !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>, ["a", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>, ["b", init] {alignment = 8 : i64}
    %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
    %3 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["__retval"] {alignment = 4 : i64}
    %4 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["result", init] {alignment = 4 : i64}
    cir.store %arg0, %0 : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
    cir.store %arg1, %1 : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>
    cir.store %arg2, %2 : !s32i, !cir.ptr<!s32i>
    %5 = cir.const #cir.fp<0.000000e+00> : !cir.float
    cir.store align(4) %5, %4 : !cir.float, !cir.ptr<!cir.float>
    cir.scope {
      %8 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
      %9 = cir.const #cir.int<0> : !s32i
      cir.store align(4) %9, %8 : !s32i, !cir.ptr<!s32i>
      %10 = cir.load align(4) %2 : !cir.ptr<!s32i>, !s32i
      cir.for : cond {
        %11 = cir.load align(4) %8 : !cir.ptr<!s32i>, !s32i
        %12 = cir.cmp(lt, %11, %10) : !s32i, !cir.bool
        cir.condition(%12)
      } body {
        cir.scope {
          %11 = cir.load align(8) %0 : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
          %12 = cir.load align(4) %8 : !cir.ptr<!s32i>, !s32i
          %13 = cir.ptr_stride(%11 : !cir.ptr<!cir.float>, %12 : !s32i), !cir.ptr<!cir.float>
          %14 = cir.load align(4) %13 : !cir.ptr<!cir.float>, !cir.float
          %15 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!cir.float>>, !cir.ptr<!cir.float>
          %16 = cir.load align(4) %8 : !cir.ptr<!s32i>, !s32i
          %17 = cir.ptr_stride(%15 : !cir.ptr<!cir.float>, %16 : !s32i), !cir.ptr<!cir.float>
          %18 = cir.load align(4) %17 : !cir.ptr<!cir.float>, !cir.float
          %19 = cir.binop(mul, %14, %18) : !cir.float
          %20 = cir.load align(4) %4 : !cir.ptr<!cir.float>, !cir.float
          %21 = cir.binop(add, %20, %19) : !cir.float
          cir.store align(4) %21, %4 : !cir.float, !cir.ptr<!cir.float>
        }
        cir.yield
      } step {
        %11 = cir.load align(4) %8 : !cir.ptr<!s32i>, !s32i
        %12 = cir.unary(inc, %11) nsw : !s32i, !s32i
        cir.store align(4) %12, %8 : !s32i, !cir.ptr<!s32i>
        cir.yield
      }
    }
    %6 = cir.load align(4) %4 : !cir.ptr<!cir.float>, !cir.float
    cir.store %6, %3 : !cir.float, !cir.ptr<!cir.float>
    %7 = cir.load %3 : !cir.ptr<!cir.float>, !cir.float
    cir.return %7 : !cir.float
  }
}

