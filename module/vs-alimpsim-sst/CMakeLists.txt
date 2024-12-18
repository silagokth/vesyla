cmake_minimum_required(VERSION 3.10)
project(vs-alimpsim-sst)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SST REQUIRED)

# Set SST include dir
execute_process(
    COMMAND sst-config --includedir
    OUTPUT_VARIABLE SST_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(SST_INCLUDE_DIR ${SST_INCLUDE_DIR})
include_directories(${SST_INCLUDE_DIR})
include_directories($ENV{SST_ELEMENTS_HOME}/include)

# Find DRRA sources and headers
file(GLOB DRRA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/components/drra/*.cpp)
file(GLOB DRRA_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/components/drra/*.h)
file(GLOB DRRA_EVENTS ${CMAKE_CURRENT_SOURCE_DIR}/components/drra/events/*.h)

# Add DRRA library
add_library(drra MODULE ${DRRA_SOURCES} ${DRRA_HEADERS} ${DRRA_EVENTS})

# Set DRRA include dirs, links and flags
target_include_directories(drra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/components/drra)
target_include_directories(drra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/components/drra/events)
target_link_options(drra PRIVATE -Wl,-undefined,dynamic_lookup)
target_link_libraries(drra SST::SSTCore)
target_compile_options(drra PRIVATE ${CMAKE_CXX_FLAGS})

# Register DRRA library and its components
message(STATUS "Registering SST library: drra")
add_custom_command(TARGET drra POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:drra> ${CMAKE_CURRENT_BINARY_DIR}/lib
    COMMAND sst-register drra drra_LIBDIR=${CMAKE_CURRENT_BINARY_DIR}/lib
)