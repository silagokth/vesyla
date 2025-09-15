find_package(VesylaUtils REQUIRED)
set(CURRENT_SUBPROJECT_NAME "vs-component")

install(DIRECTORY template/
        DESTINATION share/${PROJECT_NAME}/component_template)
