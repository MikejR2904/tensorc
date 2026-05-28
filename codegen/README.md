# Codegen Module Architecture

This directory contains the TensorC code generation system, organized into distinct subsystems for clarity and maintainability.

## Directory Structure

```
codegen/
├── legacy/              # Legacy instruction selection & emission system
│   ├── AsmPrinter.h/cpp        # RISC-V assembly output
│   ├── CodegenDriver.h/cpp      # Old pipeline orchestration
│   ├── InstrSelector.h/cpp      # IR → MachineInstr lowering
│   ├── RegAlloc.h/cpp           # Virtual → physical register allocation
│   ├── MachineInstr.h           # Machine-level instruction representation
│   ├── MachineFunction.h        # Machine-level function representation
│   ├── Target.h                 # Target architecture definitions
│   └── README.md                # Legacy system documentation
│
├── lowering/            # Progressive lowering pipeline (new system)
│   ├── LoopNest.h              # HLIR: loop hierarchy with buffers
│   ├── Tiler.h/cpp             # Phase A: HLIR → LLIR (8×8 tiling)
│   ├── ScratchpadAllocator.h/cpp   # Phase B: memory layout planning
│   ├── MemoryLegalizer.h/cpp       # Phase B: spilling & DMA injection
│   └── Scheduler.h/cpp         # Phase C: async scheduling & double-buffering
│
├── targets/             # Target-specific code emitters
│   ├── TargetEmitter.h/cpp          # Abstract base class
│   ├── X86TargetEmitter.h/cpp       # x86_64 with AVX2
│   ├── RiscVTargetEmitter.h/cpp     # RISC-V with custom .insn directives
│   └── README.md                    # Target system documentation
│
├── tools/               # Testing & diagnostic utilities
│   ├── main.cpp                 # Basic legacy pipeline test
│   ├── comprehensive_test.cpp   # Extended legacy tests
│   └── test_progressive_lowering.cpp # Progressive lowering tests
│
├── NewCodegenDriver.h/cpp   # New pipeline orchestrator
├── CMakeLists.txt
├── PROGRESSIVE_LOWERING.md  # Detailed architecture guide (400+ lines)
├── QUICKSTART.md           # Quick reference for the new system
└── README.md               # This file
```

## System Overview

### Legacy System (Functional)
- **Status**: Maintained for compatibility; used for simple operations
- **Pipeline**: IR → MachineInstr → RegAlloc → AsmPrinter → RV64 assembly
- **Location**: `legacy/`
- **Entry Point**: `CodegenDriver::lower_function_to_asm()`
- **Supported Ops**: Scalar arithmetic, memory ops, basic control flow

### Progressive Lowering Pipeline (New, Primary)
- **Status**: Main system for tensor operations
- **Architecture**: 4 phases with specialization at each level
- **Location**: `lowering/`, `targets/`, `NewCodegenDriver.*`
- **Entry Point**: `NewCodegenDriver::lower()`

#### Phase A: Tiling
- Input: High-level tensor operations (MatMul, element-wise, reductions)
- Output: Loop nests with 8×8 tiles (LLIR)
- **Component**: `Tiler`

#### Phase B: Memory Legalization
- Input: LLIR with 8 KB scratchpad constraint
- Output: Spill points, buffer allocation, DMA operations
- **Components**: `ScratchpadAllocator`, `MemoryLegalizer`
- **Key Feature**: Automatic K > 64 spilling with accumulator preservation

#### Phase C: Asynchronous Scheduling
- Input: Memory-legalized LLIR with DMA operations
- Output: Reordered compute/memory tasks with scoreboard events
- **Component**: `Scheduler`
- **Optimization**: Double-buffering for compute/memory overlap

#### Phase D: Target Emission
- Input: Scheduled LLIR
- Output: Target-specific assembly (RISC-V, x86_64)
- **Components**: `TargetEmitter` + architecture-specific subclasses

## Integration

Both systems coexist:
- **Simple ops** → Legacy system (faster iteration)
- **Tensor ops** → Progressive lowering (advanced optimization)
- **Migration**: Gradual, operation-by-operation

## Building & Testing

```bash
# Build entire codegen module
cmake --build . --target codegen

# Run legacy tests
ctest -R codegen-test

# Run comprehensive tests
ctest -R codegen-comprehensive

# Run progressive lowering tests
ctest -R codegen-progressive-test
```

## Documentation

- **PROGRESSIVE_LOWERING.md**: Comprehensive 400+ line guide covering:
  - 4-phase pipeline design
  - Memory model and scratchpad allocation
  - Automatic spilling strategy
  - Target abstraction
  - Research foundations (Halide, TVM, TPU, SCALE)

- **QUICKSTART.md**: Quick reference for using the new system

- **legacy/README.md**: Legacy system architecture

- **targets/README.md**: Target-specific emission patterns

## Key Design Principles

1. **Separation of Concerns**: Each phase has a single responsibility
2. **Target Abstraction**: Emit code without knowing specific ISA details
3. **Progressive Lowering**: Gradually specialize from high-level to low-level
4. **Preserving Semantics**: Spilling and scheduling preserve correctness
5. **Performance Optimization**: DMA pipelining, double-buffering, tiling

## Contributing

When adding new operations:
1. Define the operation in IR (compiler/ir/Instruction.h)
2. Add a tiling strategy in `Tiler`
3. Implement target emitters in `targets/`
4. Add tests in `tools/`
5. Document in PROGRESSIVE_LOWERING.md
