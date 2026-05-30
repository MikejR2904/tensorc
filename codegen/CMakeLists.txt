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

add_executable(codegen-scalar-test tools/test_legacy_scalar.cpp)
target_link_libraries(codegen-scalar-test PRIVATE codegen)

add_executable(codegen-legacy-extended-test tools/test_legacy_extended.cpp)
target_link_libraries(codegen-legacy-extended-test PRIVATE codegen)

add_executable(codegen-progressive-test tools/test_progressive_lowering.cpp)
target_link_libraries(codegen-progressive-test PRIVATE codegen)

add_custom_target(run-codegen-scalar-test
    COMMAND codegen-scalar-test
    DEPENDS codegen-scalar-test
)

add_custom_target(run-codegen-legacy-extended
    COMMAND codegen-legacy-extended-test
    DEPENDS codegen-legacy-extended-test
)

add_custom_target(run-codegen-progressive
    COMMAND codegen-progressive-test
    DEPENDS codegen-progressive-test
)


target_include_directories(codegen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(codegen PUBLIC )
