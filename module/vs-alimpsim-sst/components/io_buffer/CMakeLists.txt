get_filename_component(component_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
add_library(${component_name} OBJECT ${component_name}.cpp)
set_property(TARGET ${component_name} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${component_name} PRIVATE .)
