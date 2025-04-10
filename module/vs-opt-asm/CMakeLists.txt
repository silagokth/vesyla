cmake_minimum_required(VERSION 3.20.0)
project(iropt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(PREFIX /home/paul/Develop/llvm-project/build)
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

file(GLOB SOURCE_ASM "src/asm/*.cpp")

function(add_custom_mlir_dialect dialect dialect_namespace)
        message(STATUS "Adding custom MLIR dialect: ${dialect}")

        # Define paths - making them relative to CMAKE_BINARY_DIR
        set(DIALECT_SRC_DIR ${PROJECT_SOURCE_DIR}/src/${dialect_namespace})
        set(DIALECT_INCLUDE_DIR lib/src/${dialect_namespace})
        set(DIALECT_OUT_DIR ${CMAKE_BINARY_DIR}/${DIALECT_INCLUDE_DIR})

        # Create output directory
        file(MAKE_DIRECTORY ${DIALECT_OUT_DIR})

        set(LLVM_TARGET_DEFINITIONS ${DIALECT_SRC_DIR}/Ops.td)
        mlir_tablegen(${DIALECT_INCLUDE_DIR}/Ops.hpp.inc -gen-op-decls)
        mlir_tablegen(${DIALECT_INCLUDE_DIR}/Ops.cpp.inc -gen-op-defs)
        add_public_tablegen_target(Vesyla${dialect}OpsIncGen)

        set(LLVM_TARGET_DEFINITIONS ${DIALECT_SRC_DIR}/Dialect.td)
        mlir_tablegen(${DIALECT_INCLUDE_DIR}/Dialect.hpp.inc -gen-dialect-decls)
        mlir_tablegen(${DIALECT_INCLUDE_DIR}/Dialect.cpp.inc -gen-dialect-defs)
        add_public_tablegen_target(Vesyla${dialect}DialectIncGen)

        set(LLVM_TARGET_DEFINITIONS ${DIALECT_SRC_DIR}/Types.td)
        mlir_tablegen(${DIALECT_INCLUDE_DIR}/Types.hpp.inc -gen-typedef-decls)
        mlir_tablegen(${DIALECT_INCLUDE_DIR}/Types.cpp.inc -gen-typedef-defs)
        add_public_tablegen_target(Vesyla${dialect}TypesIncGen)

        # Create a meta-target that depends on all tablegen targets
        add_custom_target(Vesyla${dialect}IncGen DEPENDS
                Vesyla${dialect}DialectIncGen
                Vesyla${dialect}OpsIncGen
                Vesyla${dialect}TypesIncGen)

        # Add dependencies
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

add_custom_mlir_dialect(Asm asm)
add_mlir_dialect_library(VesylaAsm
        ${SOURCE_ASM}

        DEPENDS
        VesylaAsmIncGen

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
