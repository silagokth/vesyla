# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(vs_alimpsim_sst_tests)

find_package(SST REQUIRED)
find_package(GTest REQUIRED)

# Set SST include dir
execute_process(
    COMMAND sst-config --includedir
    OUTPUT_VARIABLE SST_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the source files
file(GLOB SOURCES "*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "/\\..*") # Exclude hidden dirs
list(FILTER SOURCES EXCLUDE REGEX "/build") # Exclude build dir

# Add the executable for each source file
foreach(SOURCE ${SOURCES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE})
    target_include_directories(${EXECUTABLE_NAME} PRIVATE ${SST_INCLUDE_DIR} $ENV{SST_ELEMENTS_HOME}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/common)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE GTest::GTest GTest::Main)
endforeach()