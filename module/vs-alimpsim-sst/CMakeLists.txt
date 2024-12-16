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
add_library(${PROJECT_NAME} MODULE ${CMAKE_CURRENT_SOURCE_DIR}/src/instructionEvent.h ${CMAKE_CURRENT_SOURCE_DIR}/src/drra.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Function to add a component
function(add_sst_component COMPONENT_NAME)
    # Create a module for this component
    add_library(${COMPONENT_NAME} MODULE
        ${CMAKE_CURRENT_SOURCE_DIR}/components/${COMPONENT_NAME}/${COMPONENT_NAME}.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/components/${COMPONENT_NAME}/${COMPONENT_NAME}.h
    )

    # Include directories
    target_include_directories(${COMPONENT_NAME} PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/components/${COMPONENT_NAME}
    )

    # Link
    target_link_options(${COMPONENT_NAME} PRIVATE -Wl,-undefined,dynamic_lookup)
    target_link_libraries(${COMPONENT_NAME} SST::SSTCore)

    # Add SST-specific compile and link flags
    target_compile_options(${COMPONENT_NAME} PRIVATE 
        ${CMAKE_CXX_FLAGS}
    )

    # Create an install target for this specific component
    # install(TARGETS ${COMPONENT_NAME}
    #     LIBRARY DESTINATION lib/drra-components
    # )

    # Custom install command to register this specific SST component
    # install(CODE "
    #     message(STATUS \"Registering SST component: ${COMPONENT_NAME}\")
    #     execute_process(
    #         COMMAND sst-register ${COMPONENT_NAME} ${COMPONENT_NAME}_LIBDIR=${CMAKE_INSTALL_PREFIX}/lib
    #         COMMAND sst-register SST_ELEMENT_SOURCE ${COMPONENT_NAME}=${CMAKE_CURRENT_SOURCE_DIR}/components/${COMPONENT_NAME}
    #         COMMAND sst-register SST_ELEMENT_TESTS ${COMPONENT_NAME}=${CMAKE_CURRENT_SOURCE_DIR}/components/${COMPONENT_NAME}/../tests
    #         COMMENT \"Registering SST component: ${COMPONENT_NAME}\"
    #     )
    # ")

    # Custom command to register this specific SST component
    add_custom_command(TARGET ${COMPONENT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${COMPONENT_NAME}> ${CMAKE_CURRENT_BINARY_DIR}/lib
        COMMAND sst-register ${COMPONENT_NAME} ${COMPONENT_NAME}_LIBDIR=${CMAKE_CURRENT_BINARY_DIR}/lib
        COMMAND sst-register SST_ELEMENT_SOURCE ${COMPONENT_NAME}=${CMAKE_CURRENT_SOURCE_DIR}/components/${COMPONENT_NAME}
        COMMAND sst-register SST_ELEMENT_TESTS ${COMPONENT_NAME}=${CMAKE_CURRENT_SOURCE_DIR}/components/${COMPONENT_NAME}/../tests
        COMMENT "Registering SST component: ${COMPONENT_NAME}"
        USES_TERMINAL
    )
endfunction()

# Discover and add all components
file(GLOB COMPONENT_DIRS "components/*")
foreach(COMPONENT_DIR ${COMPONENT_DIRS})
    if(IS_DIRECTORY ${COMPONENT_DIR})
        message(STATUS "Adding SST component: ${COMPONENT_DIR}")
        get_filename_component(COMPONENT_NAME ${COMPONENT_DIR} NAME)
        add_sst_component(${COMPONENT_NAME})
    endif()
endforeach()