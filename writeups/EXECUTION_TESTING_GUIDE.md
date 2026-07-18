# Execution Testing Guide

> **Superseded.** This was an aspirational design draft (the "harness"
> described below was never implemented). Real, working execution testing —
> compile real `.tcc` source, assemble and link with the local toolchain,
> execute on real hardware, compare against an independently-computed
> expected value — now exists; see
> [REAL_EXECUTION_TESTING_GUIDE.md](../REAL_EXECUTION_TESTING_GUIDE.md) at
> the repo root. This document is kept for historical reference only.

## Overview

The TensorC codegen now includes **CPU execution tests** that go beyond assembly validation. These tests:

✓ Compile generated code into executable ELF binaries  
✓ Execute directly on CPU (RISC-V emulator or native x86_64)  
✓ Verify output correctness against expected results  
✓ Catch runtime issues (memory access, register corruption, etc.)  

## Why Execution Testing?

**Assembly validation is not enough:**
- Instructions might be syntactically correct but semantically broken
- Register allocation bugs only appear at runtime
- Memory access patterns might violate alignment or bounds
- Edge cases only trigger during actual execution

**Example Problems Caught:**
```cpp
// Problem 1: Wrong register used
generated: "add x12, x13, x14"
correct:   "add x10, x10, x11"
// Assembly validator passes, but code gives wrong result!

// Problem 2: Memory corruption
// Generated code writes to wrong stack offset
// Assembly validator passes, but crashes at runtime

// Problem 3: Off-by-one in loop
// Loop counter never reaches condition
// Code hangs or produces partial results
```

## Architecture

### Test Flow

```
IR → Codegen → Assembly (.s) → Assembler → ELF Binary → Emulator/CPU → Output
                    ↓
         [Assembly Validation]     [NEW: Execution Validation]
```

### Components

1. **Code Generator**: Create assembly from IR
2. **Assembler**: Convert assembly to ELF binary (using `as`)
3. **Linker**: Link with runtime support (using `ld`)
4. **Executor**: Run binary and capture output
5. **Validator**: Compare against expected results

## Implementation

### Test Skeleton

```cpp
#include "codegen/tools/test_utils.h"
#include "codegen/tools/execution_utils.h"

void test_i64_add_execution() {
    // Step 1: Create IR
    auto fn = create_i64_add_function();
    
    // Step 2: Generate assembly
    std::string asm_code = generate_assembly(fn);
    
    // Step 3: Write assembly to file
    std::ofstream out("test_add.s");
    out << asm_code;
    out.close();
    
    // Step 4: Assemble & link
    std::string elf_path = assemble_and_link("test_add.s");
    assert(!elf_path.empty());  // Assembly must succeed
    
    // Step 5: Create test harness
    ExecutionHarness harness(elf_path);
    
    // Step 6: Execute with test cases
    int64_t result = harness.call_i64_i64_i64("test_i64_add", 10LL, 20LL);
    
    // Step 7: Validate output
    assert(result == 30LL);  // 10 + 20 = 30
    
    std::cout << "✓ test_i64_add_execution passed\n";
}
```

### Execution Utilities (execution_utils.h)

```cpp
namespace codegen {

class ExecutionHarness {
public:
    /// Load ELF binary for execution
    explicit ExecutionHarness(const std::string& elf_path);
    
    /// Call function with different signatures
    int64_t call_i64_i64_i64(const std::string& fn_name, int64_t a, int64_t b);
    double call_f64_f64_f64(const std::string& fn_name, double a, double b);
    
    /// Call with multiple arguments
    std::vector<int64_t> call_with_vector(const std::string& fn_name,
                                          const std::vector<int64_t>& args);
    
    /// Get memory region for buffer arguments
    std::vector<int64_t>& get_buffer(const std::string& name);
    
    /// Execute and get return value
    int64_t execute();
    
    /// Check for runtime errors (segfault, stack overflow, etc.)
    bool has_error() const;
    const std::string& error_message() const;
};

/// Assemble + link assembly file into ELF binary
std::string assemble_and_link(const std::string& asm_path);

/// Disassemble ELF to verify generated code
std::string disassemble_elf(const std::string& elf_path);

/// Valgrind integration: detect memory errors
bool check_memory_safety(const std::string& elf_path, 
                        int64_t a, int64_t b);

} // namespace codegen
```

## Test Cases

### 1. Scalar Arithmetic Tests

```cpp
void test_execution_suite_scalar() {
    // Basic operations
    assert_execution("i64_add", 10, 20, 30);
    assert_execution("i64_mul", 5, 6, 30);
    assert_execution("f64_add", 1.5, 2.5, 4.0);
    assert_execution("f64_mul", 2.0, 3.0, 6.0);
    
    // Edge cases
    assert_execution("i64_add", -1, 1, 0);
    assert_execution("i64_mul", 0, 999, 0);
    assert_execution("f64_add", 0.0, -0.0, 0.0);
    
    // Stress test: register pressure
    // (a+b)*(c+d) + ((a*b)+(c*d))
    assert_execution("complex_expr", 2, 3, 4, 5, 69);
}
```

### 2. Memory Access Tests

```cpp
void test_execution_memory() {
    // Load/store
    int64_t buffer[10];
    buffer[0] = 42;
    
    ExecutionHarness h("load_store.elf");
    int64_t result = h.call_with_args("test_load", buffer, 0);
    assert(result == 42);
    
    // Complex memory patterns
    // Test: load value, compute, store result
}
```

### 3. Control Flow Tests

```cpp
void test_execution_branches() {
    ExecutionHarness h("branches.elf");
    
    // if (a < b) return a else return b;
    assert(h.call_i64_i64_i64("min_func", 5, 3) == 3);
    assert(h.call_i64_i64_i64("min_func", 2, 8) == 2);
    
    // Loop: sum 1..n
    int64_t sum = h.call_i64("loop_sum", 10);  // 1+2+...+10 = 55
    assert(sum == 55);
}
```

### 4. Tensor Operation Tests

```cpp
void test_execution_tensor() {
    // Element-wise add: C[i] = A[i] + B[i]
    std::vector<double> A(64), B(64), C(64);
    for (int i = 0; i < 64; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }
    
    ExecutionHarness h("elemwise_add.elf");
    h.get_buffer("A") = convert_to_i64(A);
    h.get_buffer("B") = convert_to_i64(B);
    h.execute();
    auto result = h.get_buffer("C");
    
    // Verify: C[i] = i + i*2 = i*3
    for (int i = 0; i < 64; ++i) {
        assert(result[i] == i * 3);
    }
}
```

## Platform Support

### Linux x86_64 (Native)
- Compiler: GCC/Clang
- Assembler: GNU as
- Linker: GNU ld
- Execution: Direct binary execution

```bash
# Compile test
gcc -c test_add.s -o test_add.o
gcc -nostdlib test_add.o entry.o -o test_add
./test_add  # Returns exit code
```

### RISC-V (Emulated)
- Compiler: riscv64-linux-gnu-gcc
- Assembler: riscv64-linux-gnu-as
- Emulator: QEMU user-mode
- Execution: QEMU + syscall support

```bash
# Compile test for RISC-V
riscv64-linux-gnu-gcc -c test_add.s -o test_add.o
riscv64-linux-gnu-ld -static test_add.o -o test_add
qemu-riscv64 ./test_add  # Emulated execution
```

## Integration with Test Suite

### CMakeLists.txt Configuration

```cmake
# Enable execution tests
set(ENABLE_EXECUTION_TESTS ON)

# Platform detection
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    find_program(QEMU_RISC64 qemu-riscv64)
    if(QEMU_RISC64)
        set(RISC64_EMULATOR "${QEMU_RISC64}")
    endif()
endif()

# Add execution test target
add_executable(codegen-execution-test 
    tools/test_execution.cpp)
target_link_libraries(codegen-execution-test codegen)
```

### Running Execution Tests

```bash
# Build
cmake --build build --target codegen-execution-test

# Run
./build/bin/codegen-execution-test

# With verbose output
./build/bin/codegen-execution-test --verbose

# RISC-V execution (if available)
./build/bin/codegen-execution-test --target riscv64
```

## Error Detection

### Types of Errors Caught

1. **Correctness**: Output doesn't match expected
2. **Segmentation Fault**: Invalid memory access
3. **Illegal Instruction**: Bad opcode generated
4. **Stack Overflow**: Infinite recursion or memory overrun
5. **Timeout**: Infinite loop in generated code
6. **Register Corruption**: Caller-saved registers modified

### Example Error Output

```
Test: test_i64_add_execution
Status: FAILED

Error: Segmentation fault at 0x7fff0004
Generated code tried to write to invalid address
Stack trace:
  test_i64_add+0x40
  test_i64_add+0x50 (crash here)

Assembly at crash point:
  0x7f000050: sd x10, -8(x0)   # BAD! x0 is hardcoded zero
                               # Should be sp (x2)

Diagnostics:
  Crash address: 0x7fff0004
  Instruction: sd x10, -8(x0)
  Expected: sd x10, -8(sp)
```

## Performance Profiling

Optional: Profile execution to catch performance regressions:

```cpp
ExecutionTimer timer("test_i64_add");
timer.start();
for (int i = 0; i < 1000000; ++i) {
    h.call_i64_i64_i64("add", i, i+1);
}
timer.stop();

// Warn if slower than baseline
double baseline_ns = 2.5;  // Expected: 2.5ns per call
if (timer.avg_ns() > baseline_ns * 1.5) {
    std::cout << "WARNING: Slow execution (" 
              << timer.avg_ns() << "ns vs " 
              << baseline_ns << "ns)\n";
}
```

## Validation Checklist

Before shipping generated code:

- [ ] Assembly parsing succeeds
- [ ] Instruction mnemonics are valid
- [ ] Function structure correct (entry/exit/return)
- [ ] Register allocation complete (no unrewritten vregs)
- [ ] **[NEW]** Code executes without crash
- [ ] **[NEW]** Output matches expected results
- [ ] **[NEW]** No memory safety violations
- [ ] **[NEW]** Performance acceptable

## Future Enhancements

1. **Coverage Tracking**: Which code paths executed?
2. **Differential Testing**: Compare against reference implementation
3. **Fuzzing**: Generate random inputs to find edge cases
4. **Performance Regression**: Track benchmark times
5. **Memory Profiling**: Detect leaks/corruption early

## Troubleshooting

### "Assembler not found"
```bash
# Install binutils for your target
sudo apt-get install binutils         # x86_64
sudo apt-get install binutils-riscv64-linux-gnu  # RISC-V
```

### "QEMU not found"
```bash
# Install emulator
sudo apt-get install qemu-user-static
```

### "ELF binary is misaligned"
Check that generated code follows ABI calling conventions:
- Stack must be 16-byte aligned at function entry
- Return value in x10 (a0) for scalars
- Callee-saved registers preserved

### "Test passes with O0 but fails with O3"
Likely register allocation or scheduling issue:
- Verify liveness analysis is correct
- Check spilling decisions
- Test with `-g` flag for debug symbols

## Code Examples

### Simple Addition Test

```cpp
#include "codegen/tools/execution_utils.h"

TEST(ExecutionTests, SimpleAddition) {
    // Generate code for: int add(int a, int b) { return a + b; }
    auto fn = create_simple_add_function();
    std::string asm_code = codegen_to_string(fn);
    
    // Write and compile
    std::string elf = assemble_and_link(asm_code);
    
    // Execute
    ExecutionHarness h(elf);
    int64_t result = h.call_i64_i64_i64("add", 10, 20);
    
    // Verify
    EXPECT_EQ(result, 30);
}
```

### Tensor Execution

```cpp
TEST(ExecutionTests, MatMulExecution) {
    // Generate 32×32 matmul kernel
    auto tensor_op = create_matmul_operation(32, 32, 32);
    std::string asm_code = progressive_lower(tensor_op);
    
    // Compile and link with runtime
    std::string elf = assemble_and_link_with_runtime(asm_code);
    
    // Setup test data
    Matrix32x32 A, B, C_gold, C_actual;
    initialize_test_matrices(A, B, C_gold);
    
    // Execute
    ExecutionHarness h(elf);
    h.get_buffer("A") = A.data();
    h.get_buffer("B") = B.data();
    h.execute();
    C_actual = h.get_buffer("C");
    
    // Verify
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            EXPECT_NEAR(C_actual[i][j], C_gold[i][j], 1e-6);
        }
    }
}
```

## References

- **QEMU Documentation**: https://qemu.org/documentation/
- **GCC Assembly Documentation**: https://gcc.gnu.org/onlinedocs/
- **System V ABI**: https://github.com/hjl-tools/x86-psABI
- **RISC-V ABI**: https://github.com/riscv/riscv-elf-psabi-doc
