#/usr/bin/env bash

/opt/llvm/bin/mlir-tblgen -gen-dialect-decls Dialect.td -I /opt/llvm/include/ -I . > Dialect.hpp.inc
/opt/llvm/bin/mlir-tblgen -gen-dialect-defs Dialect.td -I /opt/llvm/include/ -I . > Dialect.cpp.inc

/opt/llvm/bin/mlir-tblgen -gen-op-decls Ops.td -I /opt/llvm/include/ -I . > Ops.hpp.inc
/opt/llvm/bin/mlir-tblgen -gen-op-defs Ops.td -I /opt/llvm/include/ -I . > Ops.cpp.inc

/opt/llvm/bin/mlir-tblgen -gen-typedef-decls Types.td -I /opt/llvm/include/ -I . > Types.hpp.inc
/opt/llvm/bin/mlir-tblgen -gen-typedef-defs Types.td -I /opt/llvm/include/ -I . > Types.cpp.inc