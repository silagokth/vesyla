cmake_minimum_required(VERSION 3.5)

project(vesyla)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC OFF)

# The project version number.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/modules/Cargo.toml" MODULES_TOML_CONTENT)
string(REGEX MATCH "version = \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\"" _ "${MODULES_TOML_CONTENT}")
set(VERSION_MAJOR ${CMAKE_MATCH_1} CACHE STRING "Project major version number.")
set(VERSION_MINOR ${CMAKE_MATCH_2} CACHE STRING "Project minor version number.")
set(VERSION_PATCH ${CMAKE_MATCH_3} CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

message(STATUS "Detected version ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH} from vs-entry")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_SHARED_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/share/${PROJECT_NAME})
set(INSTALL_ARCHIVE_OUTPUT_DIRECTORY lib)
set(INSTALL_LIBRARY_OUTPUT_DIRECTORY lib)
set(INSTALL_RUNTIME_OUTPUT_DIRECTORY bin)
set(INSTALL_SHARED_OUTPUT_DIRECTORY share/${PROJECT_NAME})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
FetchContent_MakeAvailable(Corrosion)

get_filename_component(CORROSION_TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "Target name: ${CORROSION_TARGET_NAME}")

corrosion_import_crate(
    MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/modules/Cargo.toml"
    PROFILE release
)

add_subdirectory("modules/vs-testcase")
add_subdirectory("modules/vs-schedule")
add_subdirectory("modules/vs-compile")

install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/vesyla DESTINATION bin)
install(
    DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
    DESTINATION bin
    FILES_MATCHING PATTERN "vs-*"
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_EXECUTE WORLD_READ WORLD_WRITE
)
install(DIRECTORY assets/applications DESTINATION share)
file(COPY assets/applications DESTINATION ${CMAKE_BINARY_DIR}/share)
install(DIRECTORY assets/icons DESTINATION share)
file(COPY assets/icons DESTINATION ${CMAKE_BINARY_DIR}/share)