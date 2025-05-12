find_package(VesylaUtils REQUIRED)
set(CURRENT_SUBPROJECT_NAME "vs-entry")

cargo_build(OUTPUT_EXEC_NAME vesyla)

vesyla_install(OUTPUT_EXEC_NAME vesyla)

install(DIRECTORY assets/applications DESTINATION share)
file(COPY assets/applications DESTINATION ${CMAKE_BINARY_DIR}/share)
install(DIRECTORY assets/icons DESTINATION share)
file(COPY assets/icons DESTINATION ${CMAKE_BINARY_DIR}/share)