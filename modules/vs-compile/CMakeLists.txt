cmake_minimum_required(VERSION 3.22.1)
project(vs-compile LANGUAGES CXX C)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Flex and Bison
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# Specify Flex and Bison input files
set(FLEX_INPUT src/schedule/pasm.l)
set(BISON_INPUT src/schedule/pasm.y)

# Generate lexer and parser
flex_target(PasmLexer ${FLEX_INPUT} ${CMAKE_BINARY_DIR}/PasmLexer.cpp)
bison_target(PasmParser ${BISON_INPUT} ${CMAKE_BINARY_DIR}/PasmParser.cpp)
add_flex_bison_dependency(PasmLexer PasmParser)

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

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/lib/src)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src/util/plog)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

file(GLOB SOURCE_PASM "src/pasm/*.cpp")
file(GLOB SOURCE_SCHEDULE "src/schedule/*.cpp")

function(add_custom_mlir_dialect dialect dialect_namespace)
  message(STATUS "Adding custom MLIR dialect: ${dialect}")

  # Define paths - making them relative to CMAKE_BINARY_DIR
  set(DIALECT_SRC_DIR ${PROJECT_SOURCE_DIR}/src/${dialect_namespace})
  set(DIALECT_INCLUDE_DIR ../../../build/lib/src/${dialect_namespace})
  set(DIALECT_OUT_DIR ${CMAKE_BINARY_DIR}/lib/src/${dialect_namespace})

  # print paths
  message(STATUS "Dialect source directory: ${DIALECT_SRC_DIR}")
  message(STATUS "Dialect include directory: ${DIALECT_INCLUDE_DIR}")
  message(STATUS "Dialect output directory: ${DIALECT_OUT_DIR}")

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
  add_custom_target(
    Vesyla${dialect}IncGen
    DEPENDS Vesyla${dialect}DialectIncGen Vesyla${dialect}OpsIncGen
            Vesyla${dialect}TypesIncGen Vesyla${dialect}PassesIncGen)

  # Add dependencies
  add_dependencies(mlir-headers Vesyla${dialect}IncGen)
endfunction()

add_custom_mlir_dialect(Pasm pasm)
add_mlir_dialect_library(
  VesylaPasm
  ${SOURCE_PASM}
  DISABLE_INSTALL
  DEPENDS
  VesylaPasmIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

file(GLOB SOURCES_UTIL "src/util/*.cpp" "src/util/*/*.cpp")
file(GLOB SOURCES_TM "src/tm/*.cpp")
set(CMAKE_CXX_FLAGS "-std=c++17 -pedantic -lpthread -g -O3 -fPIC")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Create object libraries
add_library(VesylaUtil ${SOURCES_UTIL})
add_library(VesylaTm ${SOURCES_TM})
add_library(VesylaSchedule ${SOURCE_SCHEDULE} ${FLEX_PasmLexer_OUTPUTS}
                           ${BISON_PasmParser_OUTPUTS})
add_dependencies(VesylaSchedule VesylaPasmIncGen) # Ensure Pasm dialect is
                                                  # generated before schedule

set(LIBS VesylaSchedule VesylaPasm VesylaTm VesylaUtil ${dialect_libs}
         ${conversion_libs})

# Add the main executable target vs-compile
add_llvm_executable(vs-compile src/exec/vs-compile/main.cpp)
llvm_update_compile_flags(vs-compile)
target_link_libraries(vs-compile PRIVATE ${LIBS})
mlir_check_all_link_libraries(vs-compile)

# Add a target to build mlir-lsp-server from util/mlir_lsp_support
add_executable(pasm-mlir-lsp-server src/exec/pasm-mlir-lsp-server/main.cpp)
target_link_libraries(pasm-mlir-lsp-server PRIVATE ${LIBS} MLIRLspServerLib)
add_dependencies(pasm-mlir-lsp-server vs-compile)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(INSTALL_GTEST
    OFF
    CACHE BOOL "" FORCE)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
# Set the unit test directory
set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unit_test)
# Automatically add all .cpp .c and .cc files in the unit test directory,
# including subdirectories This will include all files in the unit test
# directory and its subdirectories into the UNIT_TEST_SOURCES variable and add
# them to the hello_test executable
file(GLOB_RECURSE UNIT_TEST_SOURCES "${UNIT_TEST_DIR}/*.cpp"
     "${UNIT_TEST_DIR}/*.c" "${UNIT_TEST_DIR}/*.cc")
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
