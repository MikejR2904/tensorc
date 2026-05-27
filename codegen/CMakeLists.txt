add_library(codegen
    AsmPrinter.cpp
    InstrSelector.cpp
    RegAlloc.cpp
    CodegenDriver.cpp
)

add_executable(codegen-test tools/main.cpp)
target_link_libraries(codegen-test PRIVATE codegen)

add_executable(codegen-comprehensive-test tools/comprehensive_test.cpp)
target_link_libraries(codegen-comprehensive-test PRIVATE codegen)

add_custom_target(run-codegen-test
    COMMAND codegen-test
    DEPENDS codegen-test
)

add_custom_target(run-codegen-comprehensive
    COMMAND codegen-comprehensive-test
    DEPENDS codegen-comprehensive-test
)


target_include_directories(codegen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(codegen PUBLIC )
