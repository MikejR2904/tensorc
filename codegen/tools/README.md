# Codegen Testing & Diagnostics

**Location**: `codegen/tools/`

This directory contains comprehensive test suites and diagnostic utilities for the codegen module. Tests are organized by subsystem and complexity level.

## Core Test Utilities

### test_utils.h
**Purpose**: Shared testing infrastructure

**Provides**:
- `AsmInstr` - Parsed assembly instruction representation
- `parse_assembly()` - Parse assembly text into structured instructions
- `AssemblyValidator` - Validate assembly output against patterns
  - `has_mnemonic()` - Check for instruction presence
  - `has_pattern()` - Check for instruction sequences
  - `ends_with_return()` - Validate proper function termination
  - `instr_count()` - Get total instruction count
- `read_asm_file()` - Load assembly from file
- `write_reference()` - Save reference assembly

**Usage**: `#include "test_utils.h"`

## Legacy System Tests

### test_legacy_scalar.cpp
**Purpose**: Scalar operations in the legacy codegen pipeline

**Test Suite** (7 comprehensive tests):
1. **Integer Addition (i64)** - Basic add instruction verification
2. **Integer Multiplication (i64)** - mul instruction validation
3. **Floating-Point Addition (f64)** - fadd.d instruction verification
4. **Integer Division (i64)** - div/udiv instruction validation
5. **Floating-Point Multiplication (f64)** - fmul.d verification
6. **Chained Operations** - Multiple operations with data flow
7. **Register Pressure** - Stress test with many temporaries

**Validation Checks**:
- Correct instruction mnemonics present
- Proper function termination with return
- Reasonable instruction count

**Build & Run**:
```bash
cmake --build . --target codegen-scalar-test
./build/bin/codegen-scalar-test
```

### test_legacy_extended.cpp
**Purpose**: Extended operations and tensor lowering in legacy system

**Test Suite** (7 comprehensive tests):
1. **Tensor Element-Wise Add (8×8)** - Vector operations or library calls
2. **Tensor Element-Wise Multiply (8×8)** - vmul or library equivalents
3. **Matrix Multiplication (32×32)** - Library call verification
4. **Fused MatMul + ReLU (32×32)** - Kernel fusion validation
5. **Conditional Branch** - if-else logic, branch instruction generation
6. **Loop Structure** - Chained accumulation operations
7. **Mixed Type Operations** - Integer and float operations together

**Validation Checks**:
- Tensor operations generate appropriate calls or intrinsics
- Control flow branches emit conditional jump instructions
- Mixed-type operations handle both int and float correctly

**Build & Run**:
```bash
cmake --build . --target codegen-legacy-extended-test
./build/bin/codegen-legacy-extended-test
```

## Progressive Lowering Tests

### test_progressive_lowering.cpp
**Purpose**: End-to-end tests of the four-phase progressive lowering pipeline

**Test Suite** (6 comprehensive tests):
1. **MatMul Tiling (128×256×128)** - Phase A: Tiling → LLIR
   - Verifies loop structure generation
   - Checks tile dimensions and iteration bounds
   
2. **Scratchpad Allocation (8 KB)** - Phase B: Memory planning
   - Validates allocation under 8 KB constraint
   - Checks greedy reuse strategy
   
3. **K > 64 Reduction Spilling** - Phase B: Automatic spilling
   - Detects large K dimensions
   - Verifies outer loop injection for chunks
   
4. **Double-Buffering Scheduling** - Phase C: Async scheduling
   - Confirms ping/pong buffer creation
   - Validates compute/memory overlap setup
   
5. **RISC-V Target Emission** - Phase D: Target-specific code
   - Complete pipeline to RISC-V assembly
   - Validates custom .insn directives and ABI names
   
6. **x86_64 Target Emission** - Phase D: x86 alternative
   - Complete pipeline to x86 assembly
   - Validates AVX2 operations and calling conventions

**Build & Run**:
```bash
cmake --build . --target codegen-progressive-test
./build/bin/codegen-progressive-test
```

## Quick Reference: Running Tests

### Run All Tests
```bash
# Build all tests
cmake --build . --config Debug --target codegen-scalar-test codegen-legacy-extended-test codegen-progressive-test

# Run via CTest
ctest -R "codegen" --output-on-failure

# Or use convenience script
bash codegen/tools/run_tests.sh
```

### Run Individual Test Suite
```bash
# Scalar operations only
./build/bin/codegen-scalar-test

# Extended operations only
./build/bin/codegen-legacy-extended-test

# Progressive lowering only
./build/bin/codegen-progressive-test
```

### View Generated Assembly
Test programs write assembly files to the current directory:
```bash
# After running tests
ls -la test_legacy_*.s         # Legacy system outputs
ls -la test_*.s                # Progressive system outputs

# View a specific output
cat test_legacy_i64_add.s
```

## Running Tests

### Via CMake
```bash
# Build all codegen tests
cmake --build . --target codegen-test
cmake --build . --target codegen-comprehensive-test
cmake --build . --target codegen-progressive-test

# Run via CTest
ctest -R codegen-test
ctest -R codegen-comprehensive
ctest -R codegen-progressive-test

# Run with verbose output
ctest -R codegen -VV
```

### Direct Execution
```bash
# After building
./build/Debug/codegen-test
./build/Debug/codegen-comprehensive-test
./build/Debug/codegen-progressive-test

# Or on Windows
.\build\Debug\codegen-test.exe
.\build\Debug\codegen-comprehensive-test.exe
.\build\Debug\codegen-progressive-test.exe
```

## Test Architecture

Each test follows a pattern:

```cpp
// 1. Build IR
auto mod = std::make_shared<ir::IRModule>("test");
auto* fn = mod->add_function("test_fn", /* type */);
// ... populate IR ...

// 2. Lower through pipeline
codegen::NewCodegenDriver driver;
std::ostringstream asm_out;
bool success = driver.lower(fn, "riscv64", asm_out);

// 3. Validate
assert(success);
assert(/* check diagnostic */);

// 4. Inspect output (optional)
std::cout << "Generated assembly:\n" << asm_out.str() << std::endl;
```

## Debugging Tests

### Enable Diagnostic Output
Add to test code:
```cpp
driver.lower(fn, "riscv64", asm_out);
std::cerr << "Diagnostic: " << driver.last_diagnostic() << std::endl;
```

### Check Intermediate Phases
Inspect individual phases:
```cpp
// After Tiler
std::cout << "LoopNest has " << loop_nest.loops.size() << " loop levels\n";

// After ScratchpadAllocator
std::cout << "Total scratchpad used: " << total_bytes << " bytes\n";

// After Scheduler
std::cout << "Generated " << scheduled_ops.size() << " operations\n";
```

### Inspect Generated Assembly
Many tests write assembly to files:
```bash
# Find test output files
ls -la test_*.s    # RISC-V assembly
ls -la test_*.asm  # x86 assembly
```

## Adding New Tests

### For Legacy System
Add test to `comprehensive_test.cpp`:
```cpp
void test_my_operation()
{
    auto mod = std::make_shared<ir::IRModule>("test");
    // ... build IR ...
    
    // Run pipeline
    codegen::MachineFunction mf;
    // ... run InstrSelector, RegAlloc, AsmPrinter ...
    
    // Validate
    assert(/* checks */);
}

// In main():
test_my_operation();
```

### For Progressive Lowering
Add test to `test_progressive_lowering.cpp`:
```cpp
void test_my_progressive_lowering()
{
    // Build IR with tensor operations
    auto fn = build_test_function();
    
    // Run pipeline
    codegen::NewCodegenDriver driver;
    std::ostringstream asm_out;
    bool success = driver.lower(fn, "riscv64", asm_out);
    
    // Validate
    assert(success);
    assert(driver.last_diagnostic().empty() || /* expected message */);
    
    // Check output if needed
    std::string asm = asm_out.str();
    assert(asm.find("expected_instruction") != std::string::npos);
}
```

## Key Test Scenarios

### Scalar Operations
- Test basic add, mul, div operations
- Verify register allocation and spilling

### Tensor Operations (MatMul, ElemAdd)
- Test tiling with various dimensions
- Verify K > 64 detection and spilling
- Check double-buffering optimization

### Mixed Operations
- Combinations of scalar + tensor ops
- Data flow between operation types

### Edge Cases
- Zero-size dimensions
- Very large tensors (requiring spilling)
- Single-tile operations
- Dimensions not multiple of tile size (remainders)

## Continuous Integration

Tests are part of the CMake build system:
```bash
# Build everything including tests
cmake --build .

# Run all tests
ctest

# Run only codegen tests
ctest -R codegen

# Run with output
ctest --output-on-failure
```

## See Also

- **../codegen/**: Main codegen module
- **../PROGRESSIVE_LOWERING.md**: Test scenarios explained in detail
- **../CMakeLists.txt**: Build configuration
