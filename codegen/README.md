# TensorC Code Generation Module

Complete code generation infrastructure supporting both scalar and tensor operations with advanced register allocation and multi-target support.

## Quick Start

```cpp
#include "codegen/CodegenDriver.h"

CodegenDriver driver("riscv64");

// Scalar operations
driver.lower_scalar_function(fn, "output.s");

// Tensor operations
std::map<const void*, TensorShape> shapes = {{a, {128, 256, 8}}};
driver.lower_tensor_operation(matmul_op, shapes, std::cout);
```

## Architecture

Two integrated pipelines:

**Scalar Pipeline** → IR → MachineInstr → [Graph Coloring RegAlloc] → Assembly
**Tensor Pipeline** → HLIR → Tiling → Memory → Scheduling → Target Assembly

## Key Features

✓ **Graph Coloring Register Allocation** - Chaitin's algorithm with liveness analysis  
✓ **Automatic Tensor Tiling** - 8×8 systolic tiles with K>64 spilling  
✓ **Memory Optimization** - 8KB scratchpad with automatic allocation  
✓ **DMA Pipelining** - Double-buffering for compute/memory overlap  
✓ **Multi-Target** - x86_64 with AVX2, RISC-V with custom .insn  

## Directory Structure

```
codegen/
├── CodegenDriver.h/cpp      # Unified API
├── NewCodegenDriver.h/cpp   # Progressive lowering (used internally)
├── legacy/
│   ├── RegAllocGraphColoring.h/cpp  # NEW: Graph coloring allocator
│   ├── AsmPrinter, CodegenDriver, InstrSelector, etc.
├── lowering/                # 4-phase pipeline
├── targets/                 # Multi-target emitters
└── tools/                   # Tests + execution validation
```

## Detailed Guides

See [writeups/](../writeups/) for comprehensive documentation:
- **CODEGEN_ARCHITECTURE.md** - System design deep dive
- **REGISTER_ALLOCATION_GUIDE.md** - Graph coloring algorithm details
- **PROGRESSIVE_LOWERING_GUIDE.md** - 4-phase pipeline walkthrough
- **EXECUTION_TESTING_GUIDE.md** - Running generated code on CPU

## Testing

```bash
# Build tests
cmake --build build --target codegen-scalar-test codegen-legacy-extended-test \
  codegen-progressive-test codegen-execution-test

# Run
./build/bin/codegen-scalar-test                # Legacy scalar ops
./build/bin/codegen-legacy-extended-test       # Tensors + control flow
./build/bin/codegen-progressive-test           # Full 4-phase pipeline
./build/bin/codegen-execution-test             # Execute on CPU ✨
```

## What's New

- **Graph Coloring RegAlloc**: Replaced simple greedy with proper interference graph + Chaitin's algorithm
- **Execution Testing**: Verify generated code actually runs correctly on CPU
- **Unified API**: Single `CodegenDriver` entry point for both pipelines
- **Consolidated Docs**: Detailed guides in `writeups/` directory

See [CHANGELOG.md](../writeups/CHANGELOG.md) for migration notes.
