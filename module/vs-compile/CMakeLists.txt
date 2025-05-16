cmake_minimum_required(VERSION 3.20.0)
project(iropt LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

include(FetchContent)
FetchContent_Declare(lexy URL https://github.com/foonathan/lexy/releases/download/v2022.12.1/lexy-src.zip)
FetchContent_MakeAvailable(lexy)

set(LLVM_HOME $ENV{LLVM_HOME})
# print LLVM_HOME
message(STATUS "Setting LLVM_HOME: ${LLVM_HOME}")
set(MLIR_DIR ${LLVM_HOME}/lib/cmake/mlir)
set(LLVM_DIR ${LLVM_HOME}/lib/cmake/llvm)

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
file(GLOB SOURCE_PASM "src/pasm/*.cpp")

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

        set(LLVM_TARGET_DEFINITIONS ${DIALECT_SRC_DIR}/Passes.td)
        mlir_tablegen(${DIALECT_INCLUDE_DIR}/Passes.hpp.inc -gen-pass-decls)
        add_public_tablegen_target(Vesyla${dialect}PassesIncGen)

        # Create a meta-target that depends on all tablegen targets
        add_custom_target(Vesyla${dialect}IncGen DEPENDS
                Vesyla${dialect}DialectIncGen
                Vesyla${dialect}OpsIncGen
                Vesyla${dialect}TypesIncGen
                Vesyla${dialect}PassesIncGen)

        # Add dependencies
        add_dependencies(mlir-headers Vesyla${dialect}IncGen)
endfunction()

add_custom_mlir_dialect(Pasm pasm)
add_mlir_dialect_library(VesylaPasm
        ${SOURCE_PASM}

        DEPENDS
        VesylaPasmIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

file(GLOB SOURCES_UTIL "src/util/*.cpp" "src/util/*/*.cpp" "src/util/*/*.cc")
file(GLOB SOURCES_TM "src/tm/*.cpp")
set(CMAKE_CXX_FLAGS "-std=c++17 -w -pedantic -lpthread -g -O1")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(VesylaUtil ${SOURCES_UTIL})
add_library(VesylaTm ${SOURCES_TM})

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        VesylaPasm
        VesylaTm
        VesylaUtil
        foonathan::lexy
)


add_llvm_executable(iropt ${SOURCE_IROPT})

llvm_update_compile_flags(iropt)
target_link_libraries(iropt PRIVATE ${LIBS})

mlir_check_all_link_libraries(iropt)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
# Set the unit test directory
set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unit_test)
# Automatically add all .cpp .c and .cc files in the unit test directory, including subdirectories
# This will include all files in the unit test directory and its subdirectories
# into the UNIT_TEST_SOURCES variable
# and add them to the hello_test executable
file(GLOB_RECURSE UNIT_TEST_SOURCES "${UNIT_TEST_DIR}/*.cpp" "${UNIT_TEST_DIR}/*.c" "${UNIT_TEST_DIR}/*.cc")
include(GoogleTest)
# for each file in the UNIT_TEST_SOURCES variable
foreach(test_file ${UNIT_TEST_SOURCES})
  # Get the filename without the directory
  get_filename_component(test_name ${test_file} NAME_WE)
  # Create a test executable for each test file
  add_executable(${test_name} ${test_file})
  # Link the test executable with the gtest and gtest_main libraries
  target_link_libraries(${test_name} GTest::gtest GTest::gtest_main ${LIBS})
  # Add the test executable to the list of tests
  gtest_discover_tests(${test_name})
endforeach()
