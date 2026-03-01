enable_testing()
include(GoogleTest)

set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE UNIT_TEST_SOURCES "${UNIT_TEST_DIR}/*.cpp"
     "${UNIT_TEST_DIR}/*.c" "${UNIT_TEST_DIR}/*.cc")

# overall_test.cpp
get_filename_component(overall_test_name "./overall_test.cpp" NAME_WE)
add_executable(${overall_test_name} "./overall_test.cpp")
target_link_libraries(${overall_test_name} PRIVATE GTest::gtest
                                                   GTest::gtest_main ${LIBS})
gtest_discover_tests(${overall_test_name})

# affine_analyzer_test.cpp
get_filename_component(affine_test_name "./affine_analyzer_test.cpp" NAME_WE)
add_executable(${affine_test_name} "./affine_analyzer_test.cpp")
target_link_libraries(
  ${affine_test_name}
  PRIVATE ${LIBS}
          MLIROptLib
          MLIRSupport
          MLIRPass
          MLIRIR
          MLIRFuncDialect
          VesylaPasm)
