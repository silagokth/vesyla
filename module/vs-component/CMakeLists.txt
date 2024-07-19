set(CURRENT_SUBPROJECT_NAME "vs-component")

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

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_SHARED_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/share/${PROJECT_NAME})

# Find cargo and rustc
find_program(CARGO_EXECUTABLE cargo)
find_program(RUSTC_EXECUTABLE rustc)

# Assuming the Rust project is also using modules or dependencies managed by Cargo
# and the main source file is located at "src/main.rs"
file(GLOB SRC "src/*.rs")


add_custom_command(
   OUTPUT ${CMAKE_BINARY_DIR}/cargo/${CURRENT_SUBPROJECT_NAME}/release/${CURRENT_SUBPROJECT_NAME}
   DEPENDS ${SRC}
   COMMAND mkdir -p ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} &&
    mkdir -p ${CMAKE_BINARY_DIR}/cargo/${CURRENT_SUBPROJECT_NAME} &&
    cargo build --release --target-dir=${CMAKE_BINARY_DIR}/cargo/${CURRENT_SUBPROJECT_NAME} &&
    cp -p ${CMAKE_BINARY_DIR}/cargo/${CURRENT_SUBPROJECT_NAME}/release/${CURRENT_SUBPROJECT_NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(gen-${CURRENT_SUBPROJECT_NAME} ALL DEPENDS ${CMAKE_BINARY_DIR}/cargo/${CURRENT_SUBPROJECT_NAME}/release/${CURRENT_SUBPROJECT_NAME})

install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CURRENT_SUBPROJECT_NAME} DESTINATION bin COMPONENT ${CURRENT_SUBPROJECT_NAME})

install(DIRECTORY components DESTINATION share/${PROJECT_NAME})
file(COPY components DESTINATION ${CMAKE_SHARED_OUTPUT_DIRECTORY})
