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

# Specify Flex and Bison input files (parser lives under lib/Parser/)
set(FLEX_INPUT lib/Parser/pasm.l)
set(BISON_INPUT lib/Parser/pasm.y)

# Generate lexer and parser (output under this subproject's build dir so the
# build works whether vs-compile is configured standalone or as a subdirectory)
flex_target(PasmLexer ${FLEX_INPUT} ${CMAKE_CURRENT_BINARY_DIR}/PasmLexer.cpp)
bison_target(PasmParser ${BISON_INPUT} ${CMAKE_CURRENT_BINARY_DIR}/PasmParser.cpp)
add_flex_bison_dependency(PasmLexer PasmParser)

set(LLVM_HOME $ENV{LLVM_HOME})
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

# First-party public headers
include_directories(${PROJECT_SOURCE_DIR}/include)
# Conversion subsystem still lives under src/ (src-relative includes) and emits
# its tablegen output under build/lib/src/.
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMAKE_BINARY_DIR}/lib/src)
# Tablegen-generated headers (mirror include/ layout under this subproject's
# build dir so the build works whether vs-compile is configured standalone or
# as a subdirectory of a parent project)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
# Bison-generated PasmParser.hpp lands directly in this subproject's build dir
include_directories(${CMAKE_CURRENT_BINARY_DIR})
# Vendored third-party headers
include_directories(${PROJECT_SOURCE_DIR}/third_party/plog)
include_directories(${PROJECT_SOURCE_DIR}/third_party)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#==============================================================================
# Tablegen helper for a Vesyla MLIR dialect
#==============================================================================
function(add_vesyla_mlir_dialect dialect)
  message(STATUS "Adding Vesyla MLIR dialect: ${dialect}")

  set(IR_TD_DIR ${PROJECT_SOURCE_DIR}/include/vesyla/Dialect/${dialect}/IR)
  set(TR_TD_DIR ${PROJECT_SOURCE_DIR}/include/vesyla/Dialect/${dialect}/Transforms)
  set(IR_OUT_REL include/vesyla/Dialect/${dialect}/IR)
  set(TR_OUT_REL include/vesyla/Dialect/${dialect}/Transforms)

  # Ensure the output directories exist for tablegen. mlir_tablegen() resolves
  # relative paths against CMAKE_CURRENT_BINARY_DIR, so we mkdir the same.
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${IR_OUT_REL})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${TR_OUT_REL})

  # Ops
  set(LLVM_TARGET_DEFINITIONS ${IR_TD_DIR}/${dialect}Ops.td)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Ops.hpp.inc -gen-op-decls)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Ops.cpp.inc -gen-op-defs)
  add_public_tablegen_target(Vesyla${dialect}OpsIncGen)

  # Dialect
  set(LLVM_TARGET_DEFINITIONS ${IR_TD_DIR}/${dialect}Dialect.td)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Dialect.hpp.inc -gen-dialect-decls)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Dialect.cpp.inc -gen-dialect-defs)
  add_public_tablegen_target(Vesyla${dialect}DialectIncGen)

  # Types
  set(LLVM_TARGET_DEFINITIONS ${IR_TD_DIR}/${dialect}Types.td)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Types.hpp.inc -gen-typedef-decls)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Types.cpp.inc -gen-typedef-defs)
  add_public_tablegen_target(Vesyla${dialect}TypesIncGen)

  # Attrs
  set(LLVM_TARGET_DEFINITIONS ${IR_TD_DIR}/${dialect}Attrs.td)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Attrs.hpp.inc -gen-attrdef-decls -attrdefs-dialect=pasm)
  mlir_tablegen(${IR_OUT_REL}/${dialect}Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=pasm)
  add_public_tablegen_target(Vesyla${dialect}AttrsIncGen)

  # Passes
  set(LLVM_TARGET_DEFINITIONS ${TR_TD_DIR}/Passes.td)
  mlir_tablegen(${TR_OUT_REL}/Passes.hpp.inc -gen-pass-decls)
  add_public_tablegen_target(Vesyla${dialect}PassesIncGen)

  add_custom_target(
    Vesyla${dialect}IncGen
    DEPENDS Vesyla${dialect}DialectIncGen Vesyla${dialect}OpsIncGen
            Vesyla${dialect}TypesIncGen Vesyla${dialect}AttrsIncGen
            Vesyla${dialect}PassesIncGen)

  add_dependencies(mlir-headers Vesyla${dialect}IncGen)
endfunction()

add_vesyla_mlir_dialect(Pasm)

#==============================================================================
# Drra dialect: a single op (drra.rop), no custom types or attributes, so it
# only needs Ops + Dialect tablegen (not the full add_vesyla_mlir_dialect).
#==============================================================================
set(DRRA_IR_TD_DIR ${PROJECT_SOURCE_DIR}/include/vesyla/Dialect/Drra/IR)
set(DRRA_IR_OUT_REL include/vesyla/Dialect/Drra/IR)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DRRA_IR_OUT_REL})

set(LLVM_TARGET_DEFINITIONS ${DRRA_IR_TD_DIR}/DrraOps.td)
mlir_tablegen(${DRRA_IR_OUT_REL}/DrraOps.hpp.inc -gen-op-decls)
mlir_tablegen(${DRRA_IR_OUT_REL}/DrraOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(VesylaDrraOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ${DRRA_IR_TD_DIR}/DrraDialect.td)
mlir_tablegen(${DRRA_IR_OUT_REL}/DrraDialect.hpp.inc -gen-dialect-decls)
mlir_tablegen(${DRRA_IR_OUT_REL}/DrraDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(VesylaDrraDialectIncGen)

add_custom_target(
  VesylaDrraIncGen
  DEPENDS VesylaDrraOpsIncGen VesylaDrraDialectIncGen)
add_dependencies(mlir-headers VesylaDrraIncGen)

#==============================================================================
# Source globs (per layer)
#==============================================================================
file(GLOB SOURCES_DIALECT_PASM_IR        "lib/Dialect/Pasm/IR/*.cpp")
file(GLOB SOURCES_DIALECT_DRRA_IR        "lib/Dialect/Drra/IR/*.cpp")
file(GLOB SOURCES_DIALECT_PASM_TRANSFORMS
     "lib/Dialect/Pasm/Transforms/*.cpp"
     "lib/Dialect/Pasm/Transforms/ScheduleEpoch/*.cpp")
file(GLOB SOURCES_ANALYSIS_TM            "lib/Analysis/TimingModel/*.cpp")
file(GLOB SOURCES_PIPELINE               "lib/Pipeline/*.cpp")
file(GLOB SOURCES_PARSER                 "lib/Parser/*.cpp")
file(GLOB SOURCES_CODEGEN                "lib/Codegen/*.cpp")
file(GLOB SOURCES_SUPPORT                "lib/Support/*.cpp")

set(SOURCES_PASM
    ${SOURCES_DIALECT_PASM_IR}
    ${SOURCES_DIALECT_PASM_TRANSFORMS})

set(SOURCES_SCHEDULE
    ${SOURCES_PIPELINE}
    ${SOURCES_PARSER}
    ${SOURCES_CODEGEN})

set(CMAKE_CXX_FLAGS "-std=c++17 -pedantic -lpthread -g -O3 -fPIC")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#==============================================================================
# Libraries
#==============================================================================

# Pasm dialect (IR + Transforms)
add_mlir_dialect_library(
  VesylaPasm
  ${SOURCES_PASM}
  DISABLE_INSTALL
  DEPENDS
  VesylaPasmIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect)

# Drra dialect (IR only)
add_mlir_dialect_library(
  VesylaDrra
  ${SOURCES_DIALECT_DRRA_IR}
  DISABLE_INSTALL
  DEPENDS
  VesylaDrraIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect)

# Tablegen + library wiring for a pass-only directory (no dialect, just
# Passes.td next to the .cpp files).
function(add_custom_mlir_pass_dir pass_path lib_name)
  set(PASSES_SRC_DIR ${PROJECT_SOURCE_DIR}/src/${pass_path})
  set(PASSES_INCLUDE_DIR ../../../build/lib/src/${pass_path})
  set(PASSES_OUT_DIR ${CMAKE_BINARY_DIR}/lib/src/${pass_path})
  file(MAKE_DIRECTORY ${PASSES_OUT_DIR})

  set(LLVM_TARGET_DEFINITIONS ${PASSES_SRC_DIR}/Passes.td)
  mlir_tablegen(${PASSES_INCLUDE_DIR}/Passes.hpp.inc -gen-pass-decls)
  add_public_tablegen_target(${lib_name}IncGen)
  add_dependencies(mlir-headers ${lib_name}IncGen)
endfunction()

add_custom_mlir_pass_dir(conversion/affine_to_pasm VesylaAffineToPasm)
file(GLOB SOURCE_AFFINE_TO_PASM "src/conversion/affine_to_pasm/*.cpp")
add_mlir_library(
  VesylaAffineToPasm
  ${SOURCE_AFFINE_TO_PASM}
  DISABLE_INSTALL
  DEPENDS
  VesylaAffineToPasmIncGen
  VesylaPasmIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRRewrite
  MLIRAffineDialect
  VesylaPasm)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# First-party libraries
add_library(VesylaUtil ${SOURCES_SUPPORT})
add_library(VesylaTm   ${SOURCES_ANALYSIS_TM})
add_library(VesylaSchedule
            ${SOURCES_SCHEDULE}
            ${FLEX_PasmLexer_OUTPUTS}
            ${BISON_PasmParser_OUTPUTS})
add_dependencies(VesylaSchedule VesylaPasmIncGen)

set(LIBS VesylaSchedule VesylaPasm VesylaDrra VesylaAffineToPasm VesylaTm VesylaUtil
         ${dialect_libs} ${conversion_libs})

file(COPY scripts/script.py DESTINATION ${CMAKE_BINARY_DIR}/scripts)
install(FILES scripts/script.py DESTINATION scripts)
file(COPY scripts/script_icdep.py DESTINATION ${CMAKE_BINARY_DIR}/scripts)
install(FILES scripts/script_icdep.py DESTINATION scripts)
file(COPY scripts/script_grouped_no_slot0.py DESTINATION ${CMAKE_BINARY_DIR}/scripts)
install(FILES scripts/script_grouped_no_slot0.py DESTINATION scripts)

# Stage runtime config (e.g. the config holding the port table) next to the
# executable so it can be found at the default path: prog_dir()/config/.
file(COPY config/ DESTINATION ${CMAKE_BINARY_DIR}/config)
install(DIRECTORY config/ DESTINATION config)

#==============================================================================
# Executables
#==============================================================================
add_llvm_executable(vs-compile src/main.cpp)
llvm_update_compile_flags(vs-compile)
target_link_libraries(vs-compile PRIVATE ${LIBS})
mlir_check_all_link_libraries(vs-compile)

# Build the pasm MLIR LSP server
add_executable(pasm-mlir-lsp-server tools/pasm-mlir-lsp-server/main.cpp)
target_link_libraries(pasm-mlir-lsp-server PRIVATE ${LIBS} MLIRLspServerLib)
add_dependencies(pasm-mlir-lsp-server vs-compile)

#==============================================================================
# Tests (gtest unit tests)
#==============================================================================
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST
    OFF
    CACHE BOOL "" FORCE)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unit_test)
file(GLOB_RECURSE UNIT_TEST_SOURCES "${UNIT_TEST_DIR}/*.cpp"
     "${UNIT_TEST_DIR}/*.c" "${UNIT_TEST_DIR}/*.cc")
include(GoogleTest)
foreach(test_file ${UNIT_TEST_SOURCES})
  get_filename_component(test_name ${test_file} NAME_WE)
  add_executable(${test_name} ${test_file})
  target_link_libraries(${test_name} GTest::gtest GTest::gtest_main ${LIBS})
  gtest_discover_tests(${test_name})
endforeach()
