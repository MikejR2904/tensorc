# codegen/tools/CMakeLists.txt
# 
# Build configuration for codegen tools, tests, and execution harness

# ═══════════════════════════════════════════════════════════════════════════
# Builtin Module Tests
# ═══════════════════════════════════════════════════════════════════════════

if(BUILD_TESTS)
    # Test: Builtin Module Functionality Tests
    # 
    # Comprehensive tests covering:
    # - Math module (scalar functions)
    # - Tensor module (tensor operations)
    # - Element-wise operations
    # - Activation functions
    # - Linear algebra (MatMul)
    # - Reductions
    # - Integration tests (fused operations, chains)
    add_executable(test-builtin-modules
        test_builtin_modules.cpp
    )
    
    target_link_libraries(test-builtin-modules
        PRIVATE
            tensorc_lib
            gtest_main
    )
    
    target_include_directories(test-builtin-modules
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    
    # Test: Execution Examples
    # 
    # Practical examples demonstrating:
    # - Assembly pseudocode for various operations
    # - How to verify correctness on CPU
    # - Error detection and performance testing
    add_executable(test-execution-examples
        test_execution_examples.cpp
    )
    
    target_link_libraries(test-execution-examples
        PRIVATE
            tensorc_lib
            gtest_main
    )
    
    target_include_directories(test-execution-examples
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
    
    # Test: Real Execution Tests
    #
    # Compiles real TensorC *source text* through the actual compiler
    # pipeline (Lexer -> Parser -> SemanticAnalyzer -> IRBuilder ->
    # PassPipeline -> ScalarCodegenPipeline), assembles + links it with the
    # local toolchain, executes it, and compares against an independently
    # computed expected value — unlike test-builtin-modules /
    # test-execution-examples above, which compare a C++ reference
    # implementation against itself and so can't actually catch a codegen
    # bug. See REAL_EXECUTION_TESTING_GUIDE.md.
    #
    # Windows-only for now: ExecUtils.cpp uses LoadLibrary/GetProcAddress to
    # load the assembled code. A POSIX dlopen-based ExecUtils backend would
    # be a small, separate follow-up for Linux/macOS CI.
    if(WIN32)
        add_executable(test-real-execution
            test_real_execution.cpp
            ExecUtils.cpp
        )

        target_link_libraries(test-real-execution
            PRIVATE
                tensorc_lib
                codegen
                gtest_main
        )

        target_include_directories(test-real-execution
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/..
        )
    endif()

    # Execution Harness (when implemented)
    # This will handle loading and executing ELF binaries
    # Currently stubbed, full implementation in progress
    # add_library(execution-harness
    #     ExecutionHarness.cpp
    # )

    # Register all tests with GoogleTest
    include(GoogleTest)
    gtest_discover_tests(test-builtin-modules
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    gtest_discover_tests(test-execution-examples
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    if(WIN32)
        gtest_discover_tests(test-real-execution
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()

    # Create convenience target to build all tests
    add_custom_target(all-builtin-tests
        DEPENDS
            test-builtin-modules
            test-execution-examples
    )

endif()
