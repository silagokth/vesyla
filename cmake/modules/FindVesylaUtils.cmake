function(cargo_build)
  set(options)
  set(oneValueArgs OUTPUT_EXEC_NAME)
  set(multiValueArgs)

  cmake_parse_arguments(CARGO_BUILD "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  include(FetchContent)
  FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(Corrosion)

  get_filename_component(CORROSION_TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  message(STATUS "Target name: ${CORROSION_TARGET_NAME}")

  corrosion_import_crate(MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
                         PROFILE release)

  # Set the output directory for the build
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/cargo/${CORROSION_TARGET_NAME})

  if(CARGO_BUILD_OUTPUT_EXEC_NAME)
    # Add a post-build command to rename the executable
    set(ORIGINAL_EXEC_PATH $<TARGET_FILE:${CORROSION_TARGET_NAME}>)
    set(NEW_EXEC_PATH
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CARGO_BUILD_OUTPUT_EXEC_NAME}")

    add_custom_target(
      rename_${CORROSION_TARGET_NAME} ALL
      DEPENDS ${CORROSION_TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${CORROSION_TARGET_NAME}>
              "${CMAKE_BINARY_DIR}/bin/${CARGO_BUILD_OUTPUT_EXEC_NAME}"
      COMMENT
        "Copying and renaming executable to ${CARGO_BUILD_OUTPUT_EXEC_NAME}")
  else()
    add_custom_target(
      copy_${CORROSION_TARGET_NAME} ALL
      DEPENDS ${CORROSION_TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${CORROSION_TARGET_NAME}>
              "${CMAKE_BINARY_DIR}/bin/${CORROSION_TARGET_NAME}"
      COMMENT "Copying and renaming executable to ${CORROSION_TARGET_NAME}")
  endif()
endfunction()

function(vesyla_install)
  set(options)
  set(oneValueArgs OUTPUT_EXEC_NAME)
  set(multiValueArgs)

  cmake_parse_arguments(VESYLA_INSTALL "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  if(VESYLA_INSTALL_OUTPUT_EXEC_NAME)
    set(CURRENT_SUBPROJECT_NAME ${VESYLA_INSTALL_OUTPUT_EXEC_NAME})
  endif()

  add_custom_target(
    gen-${CURRENT_SUBPROJECT_NAME} ALL
    DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CURRENT_SUBPROJECT_NAME})

  install(
    PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CURRENT_SUBPROJECT_NAME}
    DESTINATION bin
    COMPONENT ${CURRENT_SUBPROJECT_NAME})
endfunction()

function(get_git_version OUT_VERSION)
  # Check if we're in a git repository
  execute_process(
    COMMAND git rev-parse --git-dir
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    RESULT_VARIABLE GIT_DIR_RESULT)

  # If not in a git repo or git command failed
  if(NOT GIT_DIR_RESULT EQUAL 0 OR "${GIT_DIR}" STREQUAL "")
    set(${OUT_VERSION}
        "version unknown"
        PARENT_SCOPE)
    return()
  endif()

  # Try to get exact tag match first
  execute_process(
    COMMAND git describe --tags --exact-match
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    RESULT_VARIABLE GIT_TAG_RESULT)

  if(GIT_TAG_RESULT EQUAL 0)
    # We're on an exact tag
    set(VERSION "${GIT_TAG}")
  else()
    # Fallback to git describe with always flag
    execute_process(
      COMMAND git describe --always --tags
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_DESCRIBE
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
      RESULT_VARIABLE GIT_DESCRIBE_RESULT)

    if(GIT_DESCRIBE_RESULT EQUAL 0)
      set(VERSION "${GIT_DESCRIBE}")
    else()
      set(VERSION "version unknown")
    endif()
  endif()

  # Make git directory path absolute if it's relative
  if(NOT IS_ABSOLUTE "${GIT_DIR}")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${GIT_DIR}")
  endif()

  # Set up configure dependencies so CMake re-runs when git state changes
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIT_DIR}/HEAD" "${GIT_DIR}/refs/heads"
             "${GIT_DIR}/refs/tags")

  set(${OUT_VERSION}
      "${VERSION}"
      PARENT_SCOPE)
endfunction()
