!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>
!u64i = !cir.int<u, 64>
!u8i = !cir.int<u, 8>
!void = !cir.void
#fn_attr = #cir<extra({nothrow = #cir.nothrow})>
#fn_attr1 = #cir<extra({inline = #cir.inline<no>, nothrow = #cir.nothrow, optnone = #cir.optnone, uwtable = #cir.uwtable<async>})>
!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E = !cir.record<class "__gnu_cxx::new_allocator<unsigned long>" padded {!u8i} #cir.record.decl.ast>
!rec_std3A3Aallocator3Cunsigned_long3E = !cir.record<class "std::allocator<unsigned long>" padded {!u8i} #cir.record.decl.ast>
!rec_std3A3Aios_base3A3AInit = !cir.record<class "std::ios_base::Init" padded {!u8i} #cir.record.decl.ast>
!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data = !cir.record<struct "std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl_data" {!cir.ptr<!u64i>, !cir.ptr<!u64i>, !cir.ptr<!u64i>} #cir.record.decl.ast>
!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl = !cir.record<struct "std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl" {!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data} #cir.record.decl.ast>
!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E = !cir.record<struct "std::_Vector_base<unsigned long, std::allocator<unsigned long>>" {!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl} #cir.record.decl.ast>
!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E = !cir.record<class "std::vector<unsigned long, std::allocator<unsigned long>>" {!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E}>
module @"<source>" attributes {cir.global_ctors = [#cir.global_ctor<"__cxx_global_var_init", 65536>], cir.lang = #cir.lang<cxx>, cir.sob = #cir.signed_overflow_behavior<undefined>, cir.triple = "x86_64-unknown-linux-gnu", cir.type_size_info = #cir.type_size_info<char = 8, int = 32, size_t = 64>, cir.uwtable = #cir.uwtable<async>, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<271> = dense<32> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e", "dlti.endianness" = "little">} {
  cir.global "private" external @__dso_handle : i8
  cir.func private @__cxa_atexit(!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>)
  cir.func private @_ZNSt8ios_base4InitC1Ev(!cir.ptr<!rec_std3A3Aios_base3A3AInit>)
  cir.func private @_ZNSt8ios_base4InitD1Ev(!cir.ptr<!rec_std3A3Aios_base3A3AInit>) extra(#fn_attr)
  cir.global "private" internal dso_local @_ZStL8__ioinit = #cir.zero : !rec_std3A3Aios_base3A3AInit {alignment = 1 : i64, ast = #cir.var.decl.ast}
  cir.func internal private @__cxx_global_var_init() {
    %0 = cir.get_global @_ZStL8__ioinit : !cir.ptr<!rec_std3A3Aios_base3A3AInit>
    cir.call @_ZNSt8ios_base4InitC1Ev(%0) : (!cir.ptr<!rec_std3A3Aios_base3A3AInit>) -> ()
    %1 = cir.get_global @_ZStL8__ioinit : !cir.ptr<!rec_std3A3Aios_base3A3AInit>
    %2 = cir.get_global @_ZNSt8ios_base4InitD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_std3A3Aios_base3A3AInit>)>>
    %3 = cir.cast(bitcast, %2 : !cir.ptr<!cir.func<(!cir.ptr<!rec_std3A3Aios_base3A3AInit>)>>), !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
    %4 = cir.cast(bitcast, %1 : !cir.ptr<!rec_std3A3Aios_base3A3AInit>), !cir.ptr<!void>
    %5 = cir.get_global @__dso_handle : !cir.ptr<i8>
    cir.call @__cxa_atexit(%3, %4, %5) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt6vectorImSaImEEC2Ev(%arg0: !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %2 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    cir.call @_ZNSt12_Vector_baseImSaImEEC2Ev(%2) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt6vectorImSaImEEixEm(%arg0: !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, %arg1: !u64i) -> !cir.ptr<!u64i> extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, ["this", init] {alignment = 8 : i64}
    %1 = cir.alloca !u64i, !cir.ptr<!u64i>, ["__n", init] {alignment = 8 : i64}
    %2 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__retval"] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>
    cir.store %arg1, %1 : !u64i, !cir.ptr<!u64i>
    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %4 = cir.base_class_addr %3 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %5 = cir.get_member %4[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %6 = cir.base_class_addr %5 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    %7 = cir.get_member %6[0] {name = "_M_start"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %8 = cir.load align(8) %7 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %9 = cir.load align(8) %1 : !cir.ptr<!u64i>, !u64i
    %10 = cir.ptr_stride(%8 : !cir.ptr<!u64i>, %9 : !u64i), !cir.ptr<!u64i>
    cir.store align(8) %10, %2 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    %11 = cir.load %2 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    cir.return %11 : !cir.ptr<!u64i>
  }
  cir.func linkonce_odr dso_local @_ZNSt6vectorImSaImEED2Ev(%arg0: !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %2 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %3 = cir.get_member %2[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %4 = cir.base_class_addr %3 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    %5 = cir.get_member %4[0] {name = "_M_start"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %6 = cir.load align(8) %5 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %7 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %8 = cir.get_member %7[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %9 = cir.base_class_addr %8 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    %10 = cir.get_member %9[1] {name = "_M_finish"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %11 = cir.load align(8) %10 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %12 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %13 = cir.call @_ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv(%12) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E> extra(#fn_attr)
    cir.try synthetic cleanup {
      cir.call exception @_ZSt8_DestroyIPmmEvT_S1_RSaIT0_E(%6, %11, %13) : (!cir.ptr<!u64i>, !cir.ptr<!u64i>, !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>) -> () cleanup {
        %15 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
        cir.call @_ZNSt12_Vector_baseImSaImEED2Ev(%15) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
        cir.yield
      }
      cir.yield
    } catch [#cir.unwind {
      cir.resume
    }]
    %14 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    cir.call @_ZNSt12_Vector_baseImSaImEED2Ev(%14) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func dso_local @_Z6dot_32v() extra(#fn_attr1) {
    %0 = cir.alloca !rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E, !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, ["a", init] {alignment = 8 : i64}
    %1 = cir.alloca !rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E, !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, ["b", init] {alignment = 8 : i64}
    %2 = cir.alloca !rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E, !cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, ["c", init] {alignment = 8 : i64}
    %3 = cir.alloca !u64i, !cir.ptr<!u64i>, ["i", init] {alignment = 8 : i64}
    cir.call @_ZNSt6vectorImSaImEEC2Ev(%0) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    cir.call @_ZNSt6vectorImSaImEEC2Ev(%1) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    cir.call @_ZNSt6vectorImSaImEEC2Ev(%2) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    %4 = cir.const #cir.int<0> : !s32i
    %5 = cir.cast(integral, %4 : !s32i), !u64i
    cir.store align(8) %5, %3 : !u64i, !cir.ptr<!u64i>
    cir.scope {
      %6 = cir.const #cir.int<0> : !s32i
      %7 = cir.cast(integral, %6 : !s32i), !u64i
      cir.store align(8) %7, %3 : !u64i, !cir.ptr<!u64i>
      cir.for : cond {
        %8 = cir.load align(8) %3 : !cir.ptr<!u64i>, !u64i
        %9 = cir.const #cir.int<32> : !s32i
        %10 = cir.cast(integral, %9 : !s32i), !u64i
        %11 = cir.cmp(lt, %8, %10) : !u64i, !cir.bool
        cir.condition(%11)
      } body {
        cir.scope {
          %8 = cir.load align(8) %3 : !cir.ptr<!u64i>, !u64i
          %9 = cir.call @_ZNSt6vectorImSaImEEixEm(%0, %8) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !u64i) -> !cir.ptr<!u64i> extra(#fn_attr)
          %10 = cir.load align(8) %9 : !cir.ptr<!u64i>, !u64i
          %11 = cir.load align(8) %3 : !cir.ptr<!u64i>, !u64i
          %12 = cir.call @_ZNSt6vectorImSaImEEixEm(%1, %11) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !u64i) -> !cir.ptr<!u64i> extra(#fn_attr)
          %13 = cir.load align(8) %12 : !cir.ptr<!u64i>, !u64i
          %14 = cir.binop(add, %10, %13) : !u64i
          %15 = cir.load align(8) %3 : !cir.ptr<!u64i>, !u64i
          %16 = cir.call @_ZNSt6vectorImSaImEEixEm(%2, %15) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !u64i) -> !cir.ptr<!u64i> extra(#fn_attr)
          cir.store align(8) %14, %16 : !u64i, !cir.ptr<!u64i>
        }
        cir.yield
      } step {
        %8 = cir.load align(8) %3 : !cir.ptr<!u64i>, !u64i
        %9 = cir.unary(inc, %8) : !u64i, !u64i
        cir.store align(8) %9, %3 : !u64i, !cir.ptr<!u64i>
        cir.yield
      }
    }
    cir.call @_ZNSt6vectorImSaImEED2Ev(%2) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    cir.call @_ZNSt6vectorImSaImEED2Ev(%1) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    cir.call @_ZNSt6vectorImSaImEED2Ev(%0) : (!cir.ptr<!rec_std3A3Avector3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Vector_baseImSaImEE12_Vector_implC2Ev(%arg0: !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>>, !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %2 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
    cir.call @_ZNSaImEC2Ev(%2) : (!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>) -> () extra(#fn_attr)
    %3 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    cir.call @_ZNSt12_Vector_baseImSaImEE17_Vector_impl_dataC2Ev(%3) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Vector_baseImSaImEEC2Ev(%arg0: !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %2 = cir.get_member %1[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    cir.call @_ZNSt12_Vector_baseImSaImEE12_Vector_implC2Ev(%2) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZN9__gnu_cxx13new_allocatorImEC2Ev(%arg0: !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>, !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSaImEC2Ev(%arg0: !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
    %2 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E> nonnull [0] -> !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>
    cir.call @_ZN9__gnu_cxx13new_allocatorImEC2Ev(%2) : (!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Vector_baseImSaImEE17_Vector_impl_dataC2Ev(%arg0: !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>>, !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    %2 = cir.get_member %1[0] {name = "_M_start"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %3 = cir.const #cir.ptr<null> : !cir.ptr<!u64i>
    cir.store align(8) %3, %2 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    %4 = cir.get_member %1[1] {name = "_M_finish"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %5 = cir.const #cir.ptr<null> : !cir.ptr<!u64i>
    cir.store align(8) %5, %4 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    %6 = cir.get_member %1[2] {name = "_M_end_of_storage"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %7 = cir.const #cir.ptr<null> : !cir.ptr<!u64i>
    cir.store align(8) %7, %6 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Destroy_auxILb1EE9__destroyIPmEEvT_S3_(%arg0: !cir.ptr<!u64i>, %arg1: !cir.ptr<!u64i>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.store %arg1, %1 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZSt8_DestroyIPmEvT_S1_(%arg0: !cir.ptr<!u64i>, %arg1: !cir.ptr<!u64i>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__first", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__last", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.store %arg1, %1 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    %2 = cir.load align(8) %0 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %3 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    cir.call @_ZNSt12_Destroy_auxILb1EE9__destroyIPmEEvT_S3_(%2, %3) : (!cir.ptr<!u64i>, !cir.ptr<!u64i>) -> ()
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZSt8_DestroyIPmmEvT_S1_RSaIT0_E(%arg0: !cir.ptr<!u64i>, %arg1: !cir.ptr<!u64i>, %arg2: !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__first", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__last", init] {alignment = 8 : i64}
    %2 = cir.alloca !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, ["", init, const] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.store %arg1, %1 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.store %arg2, %2 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>
    %3 = cir.load align(8) %0 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %4 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    cir.call @_ZSt8_DestroyIPmEvT_S1_(%3, %4) : (!cir.ptr<!u64i>, !cir.ptr<!u64i>) -> ()
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv(%arg0: !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) -> !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E> extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, ["this", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, ["__retval"] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>
    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %3 = cir.get_member %2[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %4 = cir.base_class_addr %3 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
    cir.store align(8) %4, %1 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>
    %5 = cir.load %1 : !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
    cir.return %5 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
  }
  cir.func private @_ZdlPvmSt11align_val_t(!cir.ptr<!void>, !u64i, !u64i) extra(#fn_attr)
  cir.func private @_ZdlPvm(!cir.ptr<!void>, !u64i) extra(#fn_attr)
  cir.func linkonce_odr dso_local @_ZN9__gnu_cxx13new_allocatorImE10deallocateEPmm(%arg0: !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, %arg1: !cir.ptr<!u64i>, %arg2: !u64i) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>, ["this", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__p", init] {alignment = 8 : i64}
    %2 = cir.alloca !u64i, !cir.ptr<!u64i>, ["__t", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>
    cir.store %arg1, %1 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.store %arg2, %2 : !u64i, !cir.ptr<!u64i>
    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>, !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>
    cir.scope {
      %9 = cir.const #cir.int<8> : !u64i
      %10 = cir.const #cir.int<16> : !u64i
      %11 = cir.cmp(gt, %9, %10) : !u64i, !cir.bool
      cir.if %11 {
        %12 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
        %13 = cir.cast(bitcast, %12 : !cir.ptr<!u64i>), !cir.ptr<!void>
        %14 = cir.load align(8) %2 : !cir.ptr<!u64i>, !u64i
        %15 = cir.const #cir.int<8> : !u64i
        %16 = cir.binop(mul, %14, %15) : !u64i
        %17 = cir.const #cir.int<8> : !u64i
        cir.call @_ZdlPvmSt11align_val_t(%13, %16, %17) : (!cir.ptr<!void>, !u64i, !u64i) -> () extra(#fn_attr)
        cir.return
      }
    }
    %4 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %5 = cir.cast(bitcast, %4 : !cir.ptr<!u64i>), !cir.ptr<!void>
    %6 = cir.load align(8) %2 : !cir.ptr<!u64i>, !u64i
    %7 = cir.const #cir.int<8> : !u64i
    %8 = cir.binop(mul, %6, %7) : !u64i
    cir.call @_ZdlPvm(%5, %8) : (!cir.ptr<!void>, !u64i) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt16allocator_traitsISaImEE10deallocateERS0_Pmm(%arg0: !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, %arg1: !cir.ptr<!u64i>, %arg2: !u64i) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, ["__a", init, const] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__p", init] {alignment = 8 : i64}
    %2 = cir.alloca !u64i, !cir.ptr<!u64i>, ["__n", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>
    cir.store %arg1, %1 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.store %arg2, %2 : !u64i, !cir.ptr<!u64i>
    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
    %4 = cir.base_class_addr %3 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E> nonnull [0] -> !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>
    %5 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %6 = cir.load align(8) %2 : !cir.ptr<!u64i>, !u64i
    cir.call @_ZN9__gnu_cxx13new_allocatorImE10deallocateEPmm(%4, %5, %6) : (!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, !cir.ptr<!u64i>, !u64i) -> ()
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm(%arg0: !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, %arg1: !cir.ptr<!u64i>, %arg2: !u64i) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, ["this", init] {alignment = 8 : i64}
    %1 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__p", init] {alignment = 8 : i64}
    %2 = cir.alloca !u64i, !cir.ptr<!u64i>, ["__n", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>
    cir.store %arg1, %1 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
    cir.store %arg2, %2 : !u64i, !cir.ptr<!u64i>
    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    cir.scope {
      %4 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
      %5 = cir.cast(ptr_to_bool, %4 : !cir.ptr<!u64i>), !cir.bool
      cir.if %5 {
        %6 = cir.get_member %3[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
        %7 = cir.base_class_addr %6 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
        %8 = cir.load align(8) %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
        %9 = cir.load align(8) %2 : !cir.ptr<!u64i>, !u64i
        cir.call @_ZNSt16allocator_traitsISaImEE10deallocateERS0_Pmm(%7, %8, %9) : (!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!u64i>, !u64i) -> ()
      }
    }
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Vector_baseImSaImEE12_Vector_implD2Ev(%arg0: !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>>, !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %2 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
    cir.call @_ZNSaImED2Ev(%2) : (!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSt12_Vector_baseImSaImEED2Ev(%arg0: !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>>, !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>
    %2 = cir.get_member %1[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %3 = cir.base_class_addr %2 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    %4 = cir.get_member %3[0] {name = "_M_start"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %5 = cir.load align(8) %4 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %6 = cir.get_member %1[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %7 = cir.base_class_addr %6 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    %8 = cir.get_member %7[2] {name = "_M_end_of_storage"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %9 = cir.load align(8) %8 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %10 = cir.get_member %1[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    %11 = cir.base_class_addr %10 : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl> nonnull [0] -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data>
    %12 = cir.get_member %11[0] {name = "_M_start"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl_data> -> !cir.ptr<!cir.ptr<!u64i>>
    %13 = cir.load align(8) %12 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
    %14 = cir.ptr_diff(%9, %13) : !cir.ptr<!u64i> -> !s64i
    %15 = cir.cast(integral, %14 : !s64i), !u64i
    cir.try synthetic cleanup {
      cir.call exception @_ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm(%1, %5, %15) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E>, !cir.ptr<!u64i>, !u64i) -> () cleanup {
        %17 = cir.get_member %1[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
        cir.call @_ZNSt12_Vector_baseImSaImEE12_Vector_implD2Ev(%17) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>) -> () extra(#fn_attr)
        cir.yield
      }
      cir.yield
    } catch [#cir.unwind {
      cir.resume
    }]
    %16 = cir.get_member %1[0] {name = "_M_impl"} : !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E> -> !cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>
    cir.call @_ZNSt12_Vector_baseImSaImEE12_Vector_implD2Ev(%16) : (!cir.ptr<!rec_std3A3A_Vector_base3Cunsigned_long2C_std3A3Aallocator3Cunsigned_long3E3E3A3A_Vector_impl>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZN9__gnu_cxx13new_allocatorImED2Ev(%arg0: !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>>, !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>
    cir.return
  }
  cir.func linkonce_odr dso_local @_ZNSaImED2Ev(%arg0: !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>) extra(#fn_attr1) {
    %0 = cir.alloca !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, ["this", init] {alignment = 8 : i64}
    cir.store %arg0, %0 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>, !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>
    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>>, !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E>
    %2 = cir.base_class_addr %1 : !cir.ptr<!rec_std3A3Aallocator3Cunsigned_long3E> nonnull [0] -> !cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>
    cir.call @_ZN9__gnu_cxx13new_allocatorImED2Ev(%2) : (!cir.ptr<!rec___gnu_cxx3A3Anew_allocator3Cunsigned_long3E>) -> () extra(#fn_attr)
    cir.return
  }
  cir.func private @_GLOBAL__sub_I_example.cpp() {
    cir.call @__cxx_global_var_init() : () -> ()
    cir.return
  }
}

