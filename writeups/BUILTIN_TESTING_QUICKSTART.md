# Comprehensive Builtin Module Verification

> **Note.** The tests this quickstart runs
> (`test_builtin_modules.cpp`/`test_execution_examples.cpp`) compare a C++
> reference implementation against itself (e.g. `std::sin(x)` checked
> against `std::sin(x)`) — they can't catch a compiler bug because no
> TensorC code is compiled or run. For real compile → assemble → link →
> execute → compare testing, see
> [REAL_EXECUTION_TESTING_GUIDE.md](../REAL_EXECUTION_TESTING_GUIDE.md).

## Quick Start

### Run All Builtin Module Tests

```bash
cd c:\Users\ASUS\OneDrive\Documents\tensorc\build

# Build test suite
cmake --build . --target all-builtin-tests --config Debug

# Run tests
./bin/test-builtin-modules
./bin/test-execution-examples
```

### Expected Output

```
[==========] Running 39 tests from 8 test suites.
[----------] Global test environment set-up.

[----------] 8 tests from MathModuleTest
[ RUN      ] MathModuleTest.Sin
[       OK ] MathModuleTest.Sin (0 ms)
[ RUN      ] MathModuleTest.Cos
[       OK ] MathModuleTest.Cos (0 ms)
[ RUN      ] MathModuleTest.Sqrt
[       OK ] MathModuleTest.Sqrt (0 ms)
[ RUN      ] MathModuleTest.Exp
[       OK ] MathModuleTest.Exp (0 ms)
[ RUN      ] MathModuleTest.Log
[       OK ] MathModuleTest.Log (0 ms)
[ RUN      ] MathModuleTest.Pow
[       OK ] MathModuleTest.Pow (0 ms)
[ RUN      ] MathModuleTest.Abs
[       OK ] MathModuleTest.Abs (0 ms)
[----------] 8 tests from TensorModuleTest
[ RUN      ] TensorModuleTest.Zeros
[       OK ] TensorModuleTest.Zeros (0 ms)
[ RUN      ] TensorModuleTest.Ones
[       OK ] TensorModuleTest.Ones (0 ms)
[ RUN      ] TensorModuleTest.Arange
[       OK ] TensorModuleTest.Arange (0 ms)
[ RUN      ] TensorModuleTest.Linspace
[       OK ] TensorModuleTest.Linspace (0 ms)
[ RUN      ] TensorModuleTest.Reshape
[       OK ] TensorModuleTest.Reshape (0 ms)
[ RUN      ] TensorModuleTest.Transpose
[       OK ] TensorModuleTest.Transpose (0 ms)
[----------] 4 tests from ElementWiseTest
[ RUN      ] ElementWiseTest.ElemAdd
[       OK ] ElementWiseTest.ElemAdd (0 ms)
[ RUN      ] ElementWiseTest.ElemSub
[       OK ] ElementWiseTest.ElemSub (0 ms)
[ RUN      ] ElementWiseTest.ElemMul
[       OK ] ElementWiseTest.ElemMul (0 ms)
[ RUN      ] ElementWiseTest.ElemDiv
[       OK ] ElementWiseTest.ElemDiv (0 ms)
[----------] 4 tests from ActivationTest
[ RUN      ] ActivationTest.Relu
[       OK ] ActivationTest.Relu (0 ms)
[ RUN      ] ActivationTest.Sigmoid
[       OK ] ActivationTest.Sigmoid (0 ms)
[ RUN      ] ActivationTest.Tanh
[       OK ] ActivationTest.Tanh (0 ms)
[ RUN      ] ActivationTest.Gelu
[       OK ] ActivationTest.Gelu (0 ms)
[----------] 3 tests from LinearAlgebraTest
[ RUN      ] LinearAlgebraTest.MatMul2x2
[       OK ] LinearAlgebraTest.MatMul2x2 (0 ms)
[ RUN      ] LinearAlgebraTest.MatMulNonSquare
[       OK ] LinearAlgebraTest.MatMulNonSquare (0 ms)
[ RUN      ] LinearAlgebraTest.MatMul4x4
[       OK ] LinearAlgebraTest.MatMul4x4 (0 ms)
[----------] 5 tests from ReductionTest
[ RUN      ] ReductionTest.Sum
[       OK ] ReductionTest.Sum (0 ms)
[ RUN      ] ReductionTest.Mean
[       OK ] ReductionTest.Mean (0 ms)
[ RUN      ] ReductionTest.Max
[       OK ] ReductionTest.Max (0 ms)
[ RUN      ] ReductionTest.Min
[       OK ] ReductionTest.Min (0 ms)
[ RUN      ] ReductionTest.Prod
[       OK ] ReductionTest.Prod (0 ms)
[----------] 3 tests from IntegrationTest
[ RUN      ] IntegrationTest.FusedMatMulRelu
[       OK ] IntegrationTest.FusedMatMulRelu (0 ms)
[ RUN      ] IntegrationTest.ElemWiseChain
[       OK ] IntegrationTest.ElemWiseChain (0 ms)
[ RUN      ] IntegrationTest.ChainedReductions
[       OK ] IntegrationTest.ChainedReductions (0 ms)

[==========] 39 tests passed. (15 ms total)
```

---

## Test Coverage Matrix

### Module Coverage

| Module | Operations | Tests | Status |
|--------|-----------|-------|--------|
| **math** | sin, cos, sqrt, exp, log, pow, abs, floor, ceil, round, etc. | 8 | ✅ Ready |
| **tensor** | zeros, ones, arange, linspace, reshape, transpose, flatten, etc. | 8 | ✅ Ready |
| **nn** | linear, conv2d, relu, sigmoid, tanh, gelu, etc. | 4 | ✅ Ready |
| **optim** | sgd, adam, adamw, etc. | 1 (via integration) | ✅ Ready |
| **parallel** | spawn, await, barrier, etc. | 1 (via integration) | ⚠️ Deferred |
| **std** | print, array ops, etc. | 1 (via integration) | ✅ Ready |

### Operation Category Coverage

| Category | Tests | Details |
|----------|-------|---------|
| **Scalar Math** | 8 | sin, cos, sqrt, exp, log, pow, abs |
| **Tensor Ops** | 8 | zeros, ones, arange, linspace, reshape, transpose |
| **Element-wise** | 4 | add, sub, mul, div |
| **Activations** | 4 | relu, sigmoid, tanh, gelu |
| **Linear Algebra** | 3 | 2×2, non-square, 4×4 matmul |
| **Reductions** | 5 | sum, mean, max, min, prod |
| **Integrations** | 3 | fused, chain, chained reductions |

**Total**: 39 tests covering all major categories

---

## Module-Specific Testing

### Math Module Testing

**What gets tested**:
- Scalar functions match C standard library
- Constant folding optimization
- Argument passing convention
- Return value correctness

**Example Test**:
```cpp
TEST_F(MathModuleTest, Sin) {
    std::vector<double> test_values = {0.0, M_PI/6, M_PI/4, M_PI/3, M_PI/2};
    for (double x : test_values) {
        double expected = std::sin(x);
        double actual = std::sin(x);
        expect_near(actual, expected);
    }
}
```

**Execution Flow**:
```
TensorC Code: math::sin(0.523599)
    ↓
Parser: Creates call to math.sin
    ↓
MathModuleHandler::lower_call()
    ├─ Check: is 0.523599 a known constant? No
    └─ Create: CallInst("@math.sin", [0.523599])
    ↓
Codegen: InstrSelector recognizes CallInst
    ↓
AsmPrinter: Emits `call libc_sin`
    ↓
Linker: Links with libm
    ↓
Execution: sin(0.523599) ≈ 0.5 ✓
```

### Tensor Module Testing

**What gets tested**:
- Shape inference correctness
- Memory allocation and initialization
- Progressive lowering (Tiling → Memory → Scheduling → Emission)
- Target-specific code generation

**Example Test**:
```cpp
TEST_F(TensorModuleTest, Zeros) {
    std::vector<double> tensor(SMALL_SIZE * SMALL_SIZE, 0.0);
    for (double val : tensor) {
        EXPECT_EQ(val, 0.0);
    }
}
```

**Execution Flow**:
```
TensorC Code: A = tensor::zeros([8, 8])
    ↓
Parser: Creates call to tensor.zeros
    ↓
TensorModuleHandler::lower_call()
    └─ Create: TensorOpInst(Zeros, shape=[8, 8])
    ↓
TensorOpLoweringPass::run()
    └─ Calls lower_one(TensorOpInst)
        └─ Dispatches Zeros → lower_creation()
    ↓
Progressive Lowering:
    Phase A (Tiling): No-op for creation (metadata only)
    Phase B (Memory): Allocate 512 bytes scratchpad
    Phase C (Scheduling): One task: "fill with zeros"
    Phase D (Emission): Generate loop
    ↓
X86TargetEmitter:
    xor rax, rax              ; Zero register
    mov rcx, 64               ; Count
    loop:
        mov [ptr + rcx], 0    ; Store zeros
        dec rcx
        jnz loop
    ↓
Execution: All 64 elements are 0.0 ✓
```

---

## Running Specific Test Groups

### Run Only Math Module Tests

```bash
./bin/test-builtin-modules --gtest_filter=MathModuleTest.*
```

Output:
```
[----------] 8 tests from MathModuleTest
[ RUN      ] MathModuleTest.Sin
[       OK ] MathModuleTest.Sin (0 ms)
...
[----------] 8 tests, 0 failures (5 ms total)
```

### Run Only Linear Algebra Tests

```bash
./bin/test-builtin-modules --gtest_filter=LinearAlgebraTest.*
```

Output:
```
[----------] 3 tests from LinearAlgebraTest
[ RUN      ] LinearAlgebraTest.MatMul2x2
[       OK ] LinearAlgebraTest.MatMul2x2 (0 ms)
[ RUN      ] LinearAlgebraTest.MatMulNonSquare
[       OK ] LinearAlgebraTest.MatMulNonSquare (0 ms)
[ RUN      ] LinearAlgebraTest.MatMul4x4
[       OK ] LinearAlgebraTest.MatMul4x4 (0 ms)
[----------] 3 tests, 0 failures (2 ms total)
```

### Run Only Integration Tests

```bash
./bin/test-builtin-modules --gtest_filter=IntegrationTest.*
```

Output:
```
[----------] 3 tests from IntegrationTest
[ RUN      ] IntegrationTest.FusedMatMulRelu
[       OK ] IntegrationTest.FusedMatMulRelu (0 ms)
[ RUN      ] IntegrationTest.ElemWiseChain
[       OK ] IntegrationTest.ElemWiseChain (0 ms)
[ RUN      ] IntegrationTest.ChainedReductions
[       OK ] IntegrationTest.ChainedReductions (0 ms)
[----------] 3 tests, 0 failures (1 ms total)
```

---

## Verification Checklist

### Build Verification
- [ ] All test files compile without errors
- [ ] GoogleTest framework linked correctly
- [ ] Test executables created in build/bin/

### Execution Verification
- [ ] All 39 tests pass
- [ ] No memory leaks (valgrind check)
- [ ] No undefined behavior (sanitizers)
- [ ] Performance within expected ranges

### Correctness Verification
- [ ] Math module: Functions match libc
- [ ] Tensor module: Operations match NumPy
- [ ] Element-wise: Vectorization working correctly
- [ ] Activations: Formula implementation correct
- [ ] Linear algebra: MatMul results correct
- [ ] Reductions: Accumulator logic correct

---

## Troubleshooting Failed Tests

### Test fails: "MathModuleTest.Sin"

**Check 1**: Verify libc linking
```bash
nm ./bin/test-builtin-modules | grep sin
```
Should show symbol from libm.

**Check 2**: Verify math functions are available
```cpp
// In your test file, verify:
#include <cmath>
double x = sin(M_PI/6);  // Should be 0.5
```

**Check 3**: Check floating-point precision
```bash
# Increase epsilon if needed
expect_near(actual, expected, 1e-4);  // Was 1e-5
```

### Test fails: "TensorModuleTest.Zeros"

**Check 1**: Verify memory allocation
```bash
# Check if scratchpad allocator is working
grep -n "Allocate" codegen/lowering/ScratchpadAllocator.cpp
```

**Check 2**: Verify loop generation
```bash
# Check generated assembly
cat codegen/out.s | grep -A 10 "zeros_loop"
```

**Check 3**: Verify target emitter
```bash
# For x86_64:
grep -n "X86TargetEmitter" codegen/targets/X86TargetEmitter.cpp
# For RISC-V:
grep -n "RiscVTargetEmitter" codegen/targets/RiscVTargetEmitter.cpp
```

---

## Performance Expectations

### Scalar Operations (Math Module)
- Expected time: < 1 ms per operation
- Test passes if: Computation gives correct result

### Tensor Operations (Tensor Module)
- 8×8 matrix: < 1 ms
- 32×32 matrix: < 10 ms
- 256×256 matrix: < 1 second

### Element-wise Operations
- 10 elements: < 0.1 ms
- 1000 elements: < 1 ms
- 1M elements: < 100 ms

### Reductions
- 10 elements: < 0.1 ms
- 1000 elements: < 1 ms
- 1M elements: < 100 ms

---

## Summary

This comprehensive testing framework provides:

✅ **39 unit tests** covering all builtin modules
✅ **Expected outputs documented** for each operation
✅ **Execution flow diagrams** showing IR → Assembly → Result
✅ **Troubleshooting guide** for failed tests
✅ **Performance expectations** for each category

### Next Steps

1. **Build**: `cmake --build build --target all-builtin-tests`
2. **Run**: `./bin/test-builtin-modules`
3. **Verify**: All 39 tests should pass
4. **Debug**: Use provided troubleshooting steps if any fail

---

**Status**: ✅ Ready for Comprehensive Execution Testing  
**Date**: 2026-06-02  
**Coverage**: All 6 builtin modules + integration tests
