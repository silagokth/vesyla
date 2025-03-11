set(CURRENT_SUBPROJECT_NAME "vs-component")

# Find cargo and rustc
find_program(CARGO_EXECUTABLE cargo)
find_program(RUSTC_EXECUTABLE rustc)

# Assuming the Rust project is also using modules or dependencies managed by Cargo
# and the main source file is located at "src/main.rs"
file(GLOB SRC "src/*.rs" "sst/*.j2")

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