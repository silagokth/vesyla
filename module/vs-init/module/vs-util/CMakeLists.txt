cmake_minimum_required(VERSION 3.5)

if(NOT TARGET VsUtil)

project(vs-util)

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

find_package(Boost COMPONENTS regex system filesystem unit_test_framework REQUIRED)

include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(include/vs-util)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "-std=c++17 -w -pedantic -lpthread -g -O2 -DBOOST_LOG_DYN_LINK -DELPP_NO_DEFAULT_LOG_FILE")
set(PROJECT_LINK_LIBS "stdc++fs")

file(GLOB SOURCES_VS_UTIL "src/*/*.cc" "src/*/*.cpp" "src/*.cc" "src/*.cpp")
add_library(VsUtil SHARED ${SOURCES_VS_UTIL})

install(TARGETS VsUtil LIBRARY DESTINATION lib COMPONENT vs-util)
install(DIRECTORY include/vs-util DESTINATION include)

# TEST
enable_testing()

function(add_boost_test SOURCE_FILE_NAME DEPENDENCY_LIB)
	get_filename_component(FILENAME ${SOURCE_FILE_NAME} NAME_WE)
	string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/unit_test/" "" TEST_EXECUTABLE_NAME ${SOURCE_FILE_NAME})
	string(REGEX REPLACE ".cpp" "" TEST_EXECUTABLE_NAME ${TEST_EXECUTABLE_NAME})
	string(REGEX REPLACE "/+" "_" TEST_EXECUTABLE_NAME ${TEST_EXECUTABLE_NAME})
#	string(MD5 SUFFIX ${SOURCE_FILE_NAME})
#	string(SUBSTRING ${SUFFIX} 0 8 SHORT_SUFFIX)
#	string(CONCAT TEST_EXECUTABLE_NAME ${FILENAME} _ ${SHORT_SUFFIX})
	
	add_executable(${TEST_EXECUTABLE_NAME} ${SOURCE_FILE_NAME})
	target_link_libraries(${TEST_EXECUTABLE_NAME} VsUtil ${DEPENDENCY_LIB} ${Boost_LIBRARIES} ${PROJECT_LINK_LIBS})
	set_target_properties(${TEST_EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/unit_test")
	
	file(READ "${SOURCE_FILE_NAME}" SOURCE_FILE_CONTENTS)
	string(REGEX MATCHALL "BOOST_AUTO_TEST_CASE\\( *([A-Za-z_0-9]+) *\\)" FOUND_TESTS ${SOURCE_FILE_CONTENTS})
	foreach(HIT ${FOUND_TESTS})
		string(REGEX REPLACE ".*\\( *([A-Za-z_0-9]+) *\\).*" "\\1" TEST_NAME ${HIT})
		add_test(NAME ${TEST_EXECUTABLE_NAME}.${TEST_NAME} 
							COMMAND "${CMAKE_BINARY_DIR}/unit_test/${TEST_EXECUTABLE_NAME}"
							--run_test=${TEST_NAME} --catch_system_error=yes)
	endforeach()
endfunction()

file(GLOB SOURCES_UNIT_TEST "unit_test/*.cpp" "unit_test/*/*.cpp")
foreach(F ${SOURCES_UNIT_TEST})
	add_boost_test(${F} "")
endforeach()

endif()
