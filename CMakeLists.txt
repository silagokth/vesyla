cmake_minimum_required(VERSION 3.5)

project(vesyla-suite)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC OFF)
# Show verbose command while compiling
#set(CMAKE_VERBOSE_MAKEFILE ON)

# The project version number.
set(VERSION_MAJOR 4 CACHE STRING "Project major version number.")
set(VERSION_MINOR 0 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 3 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_SHARED_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/share/${PROJECT_NAME})
set(INSTALL_ARCHIVE_OUTPUT_DIRECTORY lib)
set(INSTALL_LIBRARY_OUTPUT_DIRECTORY lib)
set(INSTALL_RUNTIME_OUTPUT_DIRECTORY bin)
set(INSTALL_SHARED_OUTPUT_DIRECTORY share/${PROJECT_NAME})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

add_subdirectory("module/vs-manas")
add_subdirectory("module/vs-alimpsim")
add_subdirectory("module/vs-component")
add_subdirectory("module/vs-testcase")
add_subdirectory("module/vs-schedule")
add_subdirectory("module/vs-entry")