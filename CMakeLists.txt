include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows: prevents GTest from overriding your compiler's runtime library settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define your test executable
add_executable(tensorc_tests 
    compiler/ast/Type.cpp 
    compiler/lexer/Lexer.cpp 
    compiler/parser/Parser.cpp 
    tests/test_irbuilder.cpp 
    # Add other source files needed for the tests
)

# Link against GTest and your compiler logic
target_link_libraries(tensorc_tests 
    GTest::gtest_main
)

# Enable the 'ctest' command
include(GoogleTest)
gtest_discover_tests(tensorc_tests)