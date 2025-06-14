set(CURRENT_SUBPROJECT_NAME "vs-schedule")

file(GLOB SRC "src/*.py")

# First check if PROTOC_PATH is defined
if(NOT DEFINED PROTOC_PATH)
  set(PROTOC_PATH "protoc")
endif()

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME}/dist/${CURRENT_SUBPROJECT_NAME}
  DEPENDS ${SRC}
  COMMAND ${PROTOC_PATH} -I=${CMAKE_CURRENT_SOURCE_DIR}/src --python_out=${CMAKE_CURRENT_SOURCE_DIR}/src ds.proto && mkdir -p ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} && mkdir -p ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME} && cd ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME} && pyinstaller -F -n ${CURRENT_SUBPROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.py && cp -p ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME}/dist/${CURRENT_SUBPROJECT_NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
)

add_custom_target(gen-${CURRENT_SUBPROJECT_NAME} ALL DEPENDS ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME}/dist/${CURRENT_SUBPROJECT_NAME})

install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CURRENT_SUBPROJECT_NAME} DESTINATION bin COMPONENT ${CURRENT_SUBPROJECT_NAME})
