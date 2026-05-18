cmake_minimum_required(VERSION 3.20...3.30)
project(TensorC LANGUAGES CXX)

# ==================== Compiler Settings ====================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build test suite" ON)

# Force all output binaries to land exactly in build/bin/ regardless of OS
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Ensure Multi-config generators (like MSVC) drop into build/bin directly instead of build/bin/Release
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} UPPER_CONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${UPPER_CONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${UPPER_CONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${UPPER_CONFIG} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# ==================== Source Modules ====================
set(CORE_SOURCES
    compiler/lexer/Lexer.cpp
    compiler/parser/Parser.cpp
    compiler/ast/Type.cpp
)

# ==================== Compiler Engine Core ====================
add_library(tensorc_lib STATIC ${CORE_SOURCES})

# Clean target inclusion layout
target_include_directories(tensorc_lib 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compiler>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Strict warnings to preserve SSA block logic integrity
target_compile_options(tensorc_lib PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror -O3>
)

# ==================== Driver Executable ====================
add_executable(tensorc cli/tensorc.cpp)
target_link_libraries(tensorc PRIVATE tensorc_lib)

# ==================== Integrated Tests ====================
if(BUILD_TESTS)
    include(FetchContent)
    
    # Secure, explicit tagged version for GoogleTest framework
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(tensorc_tests
        tests/test_ast.cpp
        tests/test_value.cpp
        tests/test_irbuilder.cpp
        tests/test_instructions.cpp
        tests/test_bb.cpp
        tests/test_function.cpp
        tests/test_irmodule.cpp
        tests/test_irprinter.cpp
    )

    target_link_libraries(tensorc_tests 
        PRIVATE 
            tensorc_lib
            gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(tensorc_tests)
endif()

# ==================== Installation Matrix ====================
install(TARGETS tensorc DESTINATION bin)

message(STATUS "TensorC build configured with C++${CMAKE_CXX_STANDARD}")