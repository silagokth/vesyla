set(CURRENT_SUBPROJECT_NAME "vs-component")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC OFF)
# Show verbose command while compiling
#set(CMAKE_VERBOSE_MAKEFILE ON)

# The project version number.
set(VERSION_MAJOR 4 CACHE STRING "Project major version number.")
set(VERSION_MINOR 0 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 0 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SRC "src/*.py")

add_custom_command(
   OUTPUT ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME}/dist/${CURRENT_SUBPROJECT_NAME}
   DEPENDS ${SRC}
   COMMAND protoc -I=${CMAKE_CURRENT_SOURCE_DIR}/src --python_out=${CMAKE_CURRENT_SOURCE_DIR}/src ds.proto && mkdir -p ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} && mkdir -p ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME} && cd ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME} && pyinstaller -F -n ${CURRENT_SUBPROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.py && cp -p ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME}/dist/${CURRENT_SUBPROJECT_NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
)

add_custom_target(gen-${CURRENT_SUBPROJECT_NAME} ALL DEPENDS ${CMAKE_BINARY_DIR}/pyinstaller/${CURRENT_SUBPROJECT_NAME}/dist/${CURRENT_SUBPROJECT_NAME})

install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CURRENT_SUBPROJECT_NAME} DESTINATION bin COMPONENT ${CURRENT_SUBPROJECT_NAME})
