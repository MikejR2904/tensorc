# Codegen Testing & Diagnostics

**Location**: `codegen/tools/`

This directory contains test programs and diagnostic utilities for the codegen module.

## Test Files

### main.cpp
**Purpose**: Basic legacy pipeline test

**Tests**:
- Simple scalar addition
- Simple scalar multiplication
- Simple branch/conditional

**Usage**:
```bash
./codegen-test
```

**Output**: Generated assembly text file for inspection

### comprehensive_test.cpp
**Purpose**: Extended tests of the legacy system

**Tests**:
- Scalar arithmetic (add, sub, mul, div with i64, f64)
- Memory operations (load, store)
- Function parameters and return values
- Control flow (branches, jumps)
- Mixed-type operations

**Coverage**:
- Tests InstrSelector, RegAlloc, AsmPrinter integration
- Exercises most code paths in the legacy pipeline
- Validates assembly output format

**Usage**:
```bash
./codegen-comprehensive-test
```

### test_progressive_lowering.cpp
**Purpose**: End-to-end tests of the new progressive lowering pipeline

**Tests**:
1. **MatMul tiling verification**
   - 32×32 MatMul → 4×4 tiles of 8×8
   - Validates tiling loop structure

2. **Scratchpad allocation**
   - 8 KB constraint respected
   - Live-range analysis works correctly

3. **K > 64 spilling detection**
   - MatMul(32×200×32) triggers automatic spilling
   - Outer reduction loops injected

4. **Double-buffering scheduler**
   - Prefetch + compute overlap detected
   - Scoreboard events inserted correctly

5. **RISC-V end-to-end emission**
   - Complete pipeline: Tiler → ScratchpadAllocator → Scheduler → RiscVTargetEmitter
   - Validates assembly output for RISC-V

6. **x86_64 end-to-end emission**
   - Complete pipeline with x86 target
   - Validates x86 assembly format

**Usage**:
```bash
./codegen-progressive-test
```

**Output**: Diagnostic information, generated assembly files

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
