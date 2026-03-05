file(GLOB SOURCES_TM "*.cpp")
add_library(VesylaTm ${SOURCES_TM})
add_dependencies(VesylaTm VesylaPasmIncGen)
target_link_libraries(VesylaTm PRIVATE MLIRPass MLIRIR MLIRFuncDialect)
