get_filename_component(last_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
get_filename_component(last_path ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(second_last_name ${last_path} NAME)
set(COMPONENT_TYPE ${second_last_name})
set(COMPONENT_NAME ${last_name})

find_package(AddComponent REQUIRED)
add_component(${COMPONENT_TYPE} ${COMPONENT_NAME})
