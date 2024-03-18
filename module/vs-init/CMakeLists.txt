cmake_minimum_required(VERSION 3.5)
project(vs-init)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC OFF)
# Show verbose command while compiling
#set(CMAKE_VERBOSE_MAKEFILE ON)

# The project version number.
set(VERSION_MAJOR 2 CACHE STRING "Project major version number.")
set(VERSION_MINOR 3 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 0 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

add_subdirectory(module/vs-util)

find_package(Boost COMPONENTS filesystem program_options REQUIRED)

include_directories(${CMAKE_BINARY_DIR})
include_directories(include)
include_directories(src)
include_directories(module/vs-util/include)

file(GLOB SOURCES_VS_INIT "src/main.cpp")

set(CMAKE_CXX_FLAGS "-std=c++17 -w -pedantic -lpthread -g -O3 -DBOOST_LOG_DYN_LINK")
set(PROJECT_LINK_LIBS "stdc++fs")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(vs-init "${SOURCES_VS_INIT}")
target_link_libraries(vs-init VsUtil ${Boost_LIBRARIES} ${PROJECT_LINK_LIBS})

install(TARGETS vs-init RUNTIME DESTINATION bin COMPONENT vs-init)
