add_library(codegen
    legacy/AsmPrinter.cpp
    legacy/InstrSelector.cpp
    legacy/RegAlloc.cpp
    legacy/CodegenDriver.cpp
    lowering/Tiler.cpp
    lowering/ScratchpadAllocator.cpp
    lowering/MemoryLegalizer.cpp
    lowering/Scheduler.cpp
    targets/TargetEmitter.cpp
    targets/X86TargetEmitter.cpp
    targets/RiscVTargetEmitter.cpp
    NewCodegenDriver.cpp
)

add_executable(codegen-test tools/main.cpp)
target_link_libraries(codegen-test PRIVATE codegen)

add_executable(codegen-comprehensive-test tools/comprehensive_test.cpp)
target_link_libraries(codegen-comprehensive-test PRIVATE codegen)

add_executable(codegen-progressive-test tools/test_progressive_lowering.cpp)
target_link_libraries(codegen-progressive-test PRIVATE codegen)

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
