function(cargo_build)
    set(options)
    set(oneValueArgs OUTPUT_EXEC_NAME)
    set(multiValueArgs)

    cmake_parse_arguments(
        CARGO_BUILD
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

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
        MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
        PROFILE release
    )

    # Set the output directory for the build
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cargo/${CORROSION_TARGET_NAME})

    if(CARGO_BUILD_OUTPUT_EXEC_NAME)
        # Add a post-build command to rename the executable
        set(ORIGINAL_EXEC_PATH $<TARGET_FILE:${CORROSION_TARGET_NAME}>)
        set(NEW_EXEC_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CARGO_BUILD_OUTPUT_EXEC_NAME}")

        add_custom_target(
            rename_${CORROSION_TARGET_NAME} ALL
            DEPENDS ${CORROSION_TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:${CORROSION_TARGET_NAME}>
            "${CMAKE_BINARY_DIR}/bin/${CARGO_BUILD_OUTPUT_EXEC_NAME}"
            COMMENT "Copying and renaming executable to ${CARGO_BUILD_OUTPUT_EXEC_NAME}"
        )
    else()
        add_custom_target(
            copy_${CORROSION_TARGET_NAME} ALL
            DEPENDS ${CORROSION_TARGET_NAME}
            COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:${CORROSION_TARGET_NAME}>
            "${CMAKE_BINARY_DIR}/bin/${CORROSION_TARGET_NAME}"
            COMMENT "Copying and renaming executable to ${CORROSION_TARGET_NAME}"
        )
    endif()
endfunction()

function(vesyla_install)
    set(options)
    set(oneValueArgs OUTPUT_EXEC_NAME)
    set(multiValueArgs)

    cmake_parse_arguments(
        VESYLA_INSTALL
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    if(VESYLA_INSTALL_OUTPUT_EXEC_NAME)
        set(CURRENT_SUBPROJECT_NAME ${VESYLA_INSTALL_OUTPUT_EXEC_NAME})
    endif()

    add_custom_target(gen-${CURRENT_SUBPROJECT_NAME} ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CURRENT_SUBPROJECT_NAME})

    install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CURRENT_SUBPROJECT_NAME} DESTINATION bin COMPONENT ${CURRENT_SUBPROJECT_NAME})
endfunction()
