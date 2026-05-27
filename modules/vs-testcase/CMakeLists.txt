find_package(VesylaUtils REQUIRED)
set(CURRENT_SUBPROJECT_NAME "vs-testcase")

file(COPY template/
     DESTINATION ${CMAKE_BINARY_DIR}/share/${PROJECT_NAME}/testcase_template)
install(DIRECTORY template/ DESTINATION share/${PROJECT_NAME}/testcase_template)
