add_library(codegen
    mir/GenericISel.cpp
    mir/PhiElimination.cpp
    mir/Liveness.cpp
    mir/LinearScanRegAlloc.cpp
    mir/CallLowering.cpp
    mir/PrologueEpilogInserter.cpp
    target/TargetInfo.cpp
    target/x86_64/X86RegisterInfo.cpp
    target/x86_64/X86InstrInfo.cpp
    target/x86_64/X86AsmPrinter.cpp
    target/x86_64/X86Target.cpp
    target/riscv64/RiscVRegisterInfo.cpp
    target/riscv64/RiscVInstrInfo.cpp
    target/riscv64/RiscVAsmPrinter.cpp
    target/riscv64/RiscVTarget.cpp
    ScalarCodegenPipeline.cpp
    lowering/Tiler.cpp
    lowering/ScratchpadAllocator.cpp
    lowering/MemoryLegalizer.cpp
    lowering/Scheduler.cpp
    targets/TargetEmitter.cpp
    targets/X86TargetEmitter.cpp
    targets/RiscVTargetEmitter.cpp
    NewCodegenDriver.cpp
    CodegenDriver.cpp
    bridge/TensorOpLowering.cpp
)

add_executable(codegen-progressive-test tools/test_progressive_lowering.cpp)
target_link_libraries(codegen-progressive-test PRIVATE codegen)

add_executable(codegen-module-bridge-test tools/test_module_bridge.cpp)
target_link_libraries(codegen-module-bridge-test PRIVATE codegen)

add_custom_target(run-codegen-progressive
    COMMAND codegen-progressive-test
    DEPENDS codegen-progressive-test
)

add_custom_target(run-codegen-module-bridge
    COMMAND codegen-module-bridge-test
    DEPENDS codegen-module-bridge-test
)

# Execution-verification test for the new scalar backend (GenericISel ->
# PhiElimination -> CallLowering -> target selection -> LinearScanRegAlloc ->
# PrologueEpilogInserter -> AsmPrinter). Replaces
# codegen-scalar-test/codegen-legacy-extended-test, which only asserted on
# assembly *text* (e.g. "contains an add mnemonic") against the now-removed
# codegen/legacy pipeline; this one actually assembles, links, and runs the
# generated x86-64 code via the local toolchain (see codegen/tools/ExecUtils.*)
# and checks real numeric results. Windows-only for now (ExecUtils.cpp uses
# LoadLibrary/GetProcAddress) — a POSIX dlopen-based ExecUtils backend would
# be a small, separate follow-up for Linux/macOS CI.
if(WIN32)
    add_executable(codegen-scalar-pipeline-test tools/test_scalar_pipeline.cpp tools/ExecUtils.cpp)
    target_link_libraries(codegen-scalar-pipeline-test PRIVATE codegen)

    add_custom_target(run-codegen-scalar-pipeline-test
        COMMAND codegen-scalar-pipeline-test
        DEPENDS codegen-scalar-pipeline-test
    )
endif()


target_include_directories(codegen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(codegen PUBLIC )
