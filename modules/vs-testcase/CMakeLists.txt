find_package(VesylaUtils REQUIRED)
set(CURRENT_SUBPROJECT_NAME "vs-testcase")

install(DIRECTORY template/ DESTINATION share/${PROJECT_NAME}/testcase_template)
