cmake_minimum_required(VERSION 3.10)
project(vs-alimpsim-sst VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SST package
find_package(SST REQUIRED)

# Set SST include dir
execute_process(
    COMMAND sst-config --includedir
    OUTPUT_VARIABLE SST_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
#set(SST_INCLUDE_DIR ${SST_INCLUDE_DIR})

# Find all component subdirectories
file(GLOB component_subdirs "components/*")
list(FILTER component_subdirs EXCLUDE REGEX "/\\..*") # Exclude hidden dirs
list(FILTER component_subdirs EXCLUDE REGEX "/build") # Exclude build dir

# Create a list to store all component libraries
set(COMPONENT_LIBRARIES)

# Add components to DRRA library
foreach(component_dir ${component_subdirs})
  if(IS_DIRECTORY ${component_dir})
    message(STATUS "Adding component: ${component_dir}")
    add_subdirectory(${component_dir})
    get_filename_component(component_name ${component_dir} NAME)
    list(APPEND COMPONENT_LIBRARIES ${component_name})
  endif()
endforeach()

# Add DRRA library
add_library(drra SHARED)

target_include_directories(drra
  INTERFACE
    ${SST_INCLUDE_DIR}
    $ENV{SST_ELEMENTS_HOME}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
)

# Link all component libraries to libdrra
#target_link_libraries(libdrra PRIVATE)

foreach(component_dir ${component_subdirs})
  if(IS_DIRECTORY ${component_dir})
    # Extract the library name from the subdirectory path
    get_filename_component(component_name ${component_dir} NAME)
    message(STATUS "Linking component: ${component_name}")
    # Check if the library target exists before linking
    if(TARGET ${component_name})
      target_include_directories(${component_name}
        PRIVATE
          ${SST_INCLUDE_DIR}
          $ENV{SST_ELEMENTS_HOME}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/include/common
      )
      target_link_libraries(drra PRIVATE ${component_name})
    else()
      message(
        WARNING "Target ${component_name} not found in ${component_dir}"
      )
    endif()
  endif()
endforeach()

# Set environment variable SST_DRRA_LIB_DIR
message(STATUS "Setting SST_DRRA_LIB_DIR to ${CMAKE_CURRENT_SOURCE_DIR}")
set(ENV{SST_DRRA_LIB_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Register DRRA library and its components
message(STATUS "Registering SST library: drra")
add_custom_command(TARGET drra POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:drra> ${CMAKE_CURRENT_BINARY_DIR}/lib
    COMMAND sst-register drra drra_LIBDIR=${CMAKE_CURRENT_BINARY_DIR}/lib
    COMMAND sst-register SST_ELEMENTS_SOURCE drra=${CMAKE_CURRENT_SOURCE_DIR}/components
    COMMAND sst-register SST_ELEMENTS_TESTS drra=${CMAKE_CURRENT_SOURCE_DIR}/components/tests
)
