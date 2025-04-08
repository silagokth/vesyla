cmake_minimum_required(VERSION 3.20.0)
project(iropt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(PREFIX /opt/llvm)
set(MLIR_DIR ${PREFIX}/lib/cmake/mlir)
set(LLVM_DIR ${PREFIX}/lib/cmake/llvm)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
include_directories(${STANDALONE_SOURCE_DIR}/src)
include_directories(${STANDALONE_BINARY_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/lib/src)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(CUSTOM_IR_DIR ${CMAKE_BINARY_DIR}/custom_ir)

file(GLOB SOURCE_CIDFG "src/cidfg/*.cpp")
file(GLOB SOURCE_IROPT "src/exec/iropt/*.cpp")

function(add_custom_mlir_dialect dialect dialect_namespace)
# print out a message
        message(STATUS "Adding custom MLIR dialect: ${dialect}")
        set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Ops.td)
        set(dialect_tblgen_deps ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Dialect.td
        ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Ops.td
        ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Types.td)
        set(dialect_tblgen_outputs ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Dialect.hpp.inc
                ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Dialect.cpp.inc
                ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Ops.hpp.inc
                ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Ops.cpp.inc
                ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Types.hpp.inc
                ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Types.cpp.inc)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Dialect.hpp.inc 
                   COMMAND mkdir -p ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace} &&
                   ${PREFIX}/bin/mlir-tblgen -gen-dialect-decls ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Dialect.td -I ${PREFIX}/include/ -I ${PROJECT_SOURCE_DIR}/src/ > ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Dialect.hpp.inc &&
                   ${PREFIX}/bin/mlir-tblgen -gen-dialect-defs ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Dialect.td -I ${PREFIX}/include/ -I ${PROJECT_SOURCE_DIR}/src/ > ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Dialect.cpp.inc &&
                   ${PREFIX}/bin/mlir-tblgen -gen-op-decls ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Ops.td -I ${PREFIX}/include/ -I ${PROJECT_SOURCE_DIR}/src/ > ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Ops.hpp.inc &&
                   ${PREFIX}/bin/mlir-tblgen -gen-op-defs ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Ops.td -I ${PREFIX}/include/ -I ${PROJECT_SOURCE_DIR}/src/ > ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Ops.cpp.inc &&
                   ${PREFIX}/bin/mlir-tblgen -gen-typedef-decls ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Types.td -I ${PREFIX}/include/ -I ${PROJECT_SOURCE_DIR}/src/ > ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Types.hpp.inc &&
                   ${PREFIX}/bin/mlir-tblgen -gen-typedef-defs ${PROJECT_SOURCE_DIR}/src/${dialect_namespace}/Types.td -I ${PREFIX}/include/ -I ${PROJECT_SOURCE_DIR}/src/ > ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace}/Types.cpp.inc
                   DEPENDS ${dialect_tblgen_deps})
        message(STATUS "TableGen dependencies: ${dialect_tblgen_outputs}")
        set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${dialect_tblgen_outputs})
        message(STATUS "TableGen output: ${TABLEGEN_OUTPUT}")
        add_public_tablegen_target(Vesyla${dialect}IncGen)
        add_dependencies(mlir-headers Vesyla${dialect}IncGen)
endfunction()

add_custom_mlir_dialect(Cidfg cidfg)

add_mlir_dialect_library(VesylaCidfg
        ${SOURCE_CIDFG}
        
        DEPENDS
        VesylaCidfgIncGen
        
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        VesylaCidfg
        )
add_llvm_executable(iropt ${SOURCE_IROPT})

llvm_update_compile_flags(iropt)
target_link_libraries(iropt PRIVATE ${LIBS})

mlir_check_all_link_libraries(iropt)





