# TensorC Code Generation Architecture

## System Design

The TensorC compiler implements a **dual-pipeline code generation architecture** optimized for both scalar and tensor operations:

```
                ┌─────────────────────────────────────────────┐
                │         IR Module (High-Level)              │
                │    (Scalar Instructions + Tensor Ops)       │
                └────────────────┬────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
        [Scalar Operations]            [Tensor Operations]
                │                                 │
        ┌───────▼───────┐              ┌────────▼─────────┐
        │  Legacy       │              │  Progressive     │
        │  Pipeline     │              │  Lowering        │
        └───────┬───────┘              └────────┬─────────┘
                │                                 │
        ┌───────▼───────────────────────────────┐│
        │  CodegenDriver (Unified Entry Point)  ││
        └───────┬───────────────────────────────┘│
                │                                 │
        ┌───────▼──────────────────┐    ┌───────▼──────────────┐
        │  Legacy Pipeline:         │    │ Progressive Pipeline │
        │  IR → MachineInstr →      │    │ (4 phases):         │
        │  RegAlloc (GC) → Assembly │    │ A: Tiling           │
        │                           │    │ B: Memory           │
        └───────────────────────────┘    │ C: Scheduling       │
                                         │ D: Emission         │
                                         └─────┬──────────────┘
                                               │
                        ┌──────────────────────┴──────────────────┐
                        │                                         │
                ┌───────▼────────┐                    ┌──────────▼─────┐
                │ x86_64 Emitter │                    │ RISC-V Emitter  │
                │ (AVX2 SIMD)    │                    │ (RV64, Custom)  │
                └────────────────┘                    └─────────────────┘
                        │                                         │
                ┌───────▼──────────────────────────────────────▼─────┐
                │         Assembly Code (.s files)                   │
                └───────┬──────────────────────────────────────┬─────┘
                        │                                      │
                ┌───────▼─────────────┐            ┌──────────▼────────┐
                │ GNU Assembler (as)  │            │ Emulator/CPU      │
                │ + Linker (ld)       │            │ (x86_64 native /  │
                └───────┬─────────────┘            │  QEMU RISC-V)     │
                        │                          └───────────────────┘
                ┌───────▼─────────────┐
                │ ELF Binary Artifact │
                └─────────────────────┘
                        │
                ┌───────▼─────────────────────────┐
                │ Execution + Validation          │
                │ ✓ Correctness checks           │
                │ ✓ Memory safety verification   │
                │ ✓ Performance profiling        │
                └─────────────────────────────────┘
```

## 1. Scalar Pipeline (Legacy)

### Purpose
Compile simple IR operations (arithmetic, memory, control flow) to efficient assembly.

### Data Flow
```
ir::Instruction (scalar)
    ↓
    InstrSelector.visit()
    ↓
MachineInstr (vregs)
    ↓
    RegAllocGraphColoring.allocate()
    ↓
MachineInstr (physical regs)
    ↓
    AsmPrinter.print()
    ↓
RV64 Assembly Text
```

### Key Components

#### MachineInstr Representation
- **MachineInstr**: Opcode + operands (registers, immediates, memory)
- **MachineOperand**: Represents a single operand
  - Physical register: `reg_num = 10` (x10/a0)
  - Virtual register: `vreg = 100` (placeholder, allocated later)
  - Immediate: `imm = 42`
  - Memory: `[base_reg + offset]`

#### InstrSelector (IR → MachineInstr)
Visitor pattern over IR instructions:

```cpp
class InstrSelector : public ir::Visitor {
    void visit(ir::BinOpInst& op) {
        // Create machine instruction for binary operation
        // Handle type coercion (i32/i64/f32/f64)
    }
    
    void visit(ir::CmpInst& cmp) {
        // Create comparison + branch setup
        // Map ir::CmpCode to machine opcodes (blt, beq, etc.)
    }
    
    void visit(ir::CondBranchInst& br) {
        // Create conditional branch instruction
    }
    
    // ... more visitors
};
```

**Supported Operations:**
- Arithmetic: add, sub, mul, div (int & float)
- Comparison: eq, ne, lt, le, gt, ge
- Memory: load, store (1-64 byte values)
- Control: branch, conditional branch, call, return
- Tensor pseudo-ops: calls to library functions

#### Register Allocation with Graph Coloring

See [REGISTER_ALLOCATION_GUIDE.md](REGISTER_ALLOCATION_GUIDE.md) for details.

**Key improvements over greedy:**
- Liveness analysis (true dataflow)
- Interference graph (actual conflicts)
- Chaitin's algorithm (optimal coloring)
- Intelligent spilling (lowest-cost vregs first)

#### AsmPrinter (MachineInstr → RV64 Assembly)
Converts machine instructions to GNU assembler syntax:

```cpp
class AsmPrinter {
    void print_instruction(const MachineInstr& mi) {
        // Emit: "  opcode  op0, op1, op2"
        // Convert register numbers to ABI names (x10 → a0)
        // Emit GAS directives (.text, .global, etc.)
    }
};
```

**Output format:**
```asm
.text
.global my_function
my_function:
  addi sp, sp, -16
  sd ra, 0(sp)
  
  # Actual code here
  add a0, a1, a2
  mul a3, a3, a4
  
  ld ra, 0(sp)
  addi sp, sp, 16
  ret
```

## 2. Progressive Lowering Pipeline

### Purpose
Compile tensor operations with sophisticated optimization:
- Automatic tiling for systolic arrays
- Memory management (8KB scratchpad constraint)
- Automatic K>64 spilling
- DMA pipelining with double-buffering

### 4-Phase Architecture

#### Phase A: HLIR → LLIR (Tiling)

**Input**: `ir::TensorOpInst` with operation code (MatMul, ElemAdd, etc.)
**Output**: Loop nest with 8×8 tiles

**Process:**
```cpp
LoopNest lower(ir::TensorOpInst* op) {
    // Extract operation semantics
    // Generate tiled loop structure
    // Create buffer regions (InputA, InputB, Accumulator, Temp)
    
    return LoopNest{
        dims: [i_tile, j_tile, k_tile, ...],
        buffers: [bufferA, bufferB, bufferAcc, ...],
        compute_blocks: [systolic_matmul, ...],
        requires_spilling: false  // or true if K > 64
    };
}
```

**Example: 128×256 MatMul with 8×8 tiles:**
```
for i = 0 to 128 step 8
  for j = 0 to 256 step 8
    for k = 0 to 256 step 64  // K dimension
      Compute C[i:i+8, j:j+8] += A[i:i+8, k:k+64] * B[k:k+64, j:j+8]
      if k+64 >= 256 and K > 64  // Need spilling
        Spill partial accumulator to temporary buffer
```

#### Phase B: Legalization & Memory Allocation

**Input**: LLIR (loop nest + buffers)
**Output**: DMA operations + buffer addresses

**Components:**
1. **ScratchpadAllocator**: Assign buffer addresses in 8KB scratchpad
   ```
   InputA:    offset=0,    size=4096 bytes
   InputB:    offset=4096, size=2048 bytes
   Accumulator: offset=6144, size=1024 bytes
   Temporary: offset=7168, size=1024 bytes
   ```

2. **MemoryLegalizer**: Insert DMA operations
   ```
   DMA_Read(main_memory → scratchpad InputA)
   DMA_Read(main_memory → scratchpad InputB)
   Compute(with local buffers)
   DMA_Write(scratchpad Accumulator → main_memory)
   ```

3. **Automatic Spilling**: If K > 64
   ```
   // Intermediate accumulator results saved to scratchpad
   // Restored before next K chunk
   ```

#### Phase C: Scheduling & Double-Buffering

**Input**: Legalized LLIR with DMA operations
**Output**: Reordered operations for overlap

**Scheduler creates:**
- DMA event dependencies
- Scoreboard for tracking in-flight operations
- Double-buffering of input buffers
  ```
  Ping buffer: DMA iteration 0
  Pong buffer: DMA iteration 1
  Compute: uses Ping
  ```

**Expected improvement:** ~90% utilization (compute + memory overlap)

#### Phase D: Target Emission

**Input**: Scheduled MLIR
**Output**: Assembly for specific target

**Multi-target support:**
```cpp
class TargetEmitter {
    virtual void emit_dma_read(const DMAOp&) = 0;
    virtual void emit_compute(const ComputeOp&) = 0;
    virtual void emit_dma_write(const DMAOp&) = 0;
};

class X86TargetEmitter : public TargetEmitter {
    // x86_64 with AVX2 SIMD
    void emit_dma_read(const DMAOp&) {
        // memcpy simulation or actual DMA instructions
    }
};

class RiscVTargetEmitter : public TargetEmitter {
    // RISC-V with custom .insn directives
    void emit_dma_read(const DMAOp&) {
        // Custom tensor instructions
    }
};
```

## 3. Unified CodegenDriver

Entry point for both pipelines:

```cpp
class CodegenDriver {
public:
    /// Scalar operations
    bool lower_scalar_function(ir::Function* fn, const std::string& out_path);
    
    /// Tensor operations
    bool lower_tensor_operation(ir::TensorOpInst* op, 
                                const std::map<const void*, TensorShape>& shapes,
                                std::ostream& out);
};
```

**Dispatch logic:**
```cpp
// Internal: determines which pipeline to use
if (ir::TensorOpInst* tensor_op = dynamic_cast<ir::TensorOpInst*>(instr)) {
    // Use progressive lowering pipeline
    return progressive_pipeline_->lower(tensor_op, shapes, out);
} else if (ir::Function* fn = ...) {
    // Use legacy pipeline
    return legacy_pipeline_->lower(fn, out_path);
}
```

## 4. Target Architecture Support

### x86_64 with AVX2

**Registers:**
- GPR: rax, rbx, rcx, rdx, rsi, rdi, r8-r15
- XMM: xmm0-xmm15 (128-bit SSE)
- YMM: ymm0-ymm15 (256-bit AVX2)

**SIMD Instructions:**
- `vmovapd ymm0, [rsi]` - Load aligned double
- `vmulpd ymm0, ymm0, ymm1` - Multiply packed doubles
- `vaddpd ymm2, ymm0, ymm1` - Add packed doubles

**Calling Convention (System V AMD64 ABI):**
- Return value: rax (or rdx:rax for 128-bit)
- Arguments: rdi, rsi, rdx, rcx, r8, r9
- Callee-saved: rbx, rsp, rbp, r12-r15

### RISC-V 64-bit

**Registers:**
- GPR: x0-x31 (x10-x17 for temporaries)
- FPR: f0-f31 (f10-f17 for temporaries)

**Vector Extensions (V):**
- Custom `.insn` directives for tensor operations
- Example: `.insn r 0x3b, 7, 5, x1, x2, x3` (systolic matmul)

**Calling Convention (RISC-V ABI):**
- Return value: a0 (x10)
- Arguments: a0-a7 (x10-x17)
- Callee-saved: s0-s11 (x8, x9, x18-x27)

## 5. Error Handling & Diagnostics

### Error Types
1. **Unsupported Operation**: Operation not implemented in any pipeline
2. **Memory Constraint Violated**: Exceeds 8KB scratchpad
3. **Compilation Error**: Invalid IR, type mismatch
4. **Assembly Error**: Bad mnemonics or syntax

### Diagnostics Output
```
Error: Matrix multiply cannot fit in scratchpad
  Requested: 12KB (InputA: 4KB + InputB: 4KB + Accum: 2KB + Temp: 2KB)
  Available: 8KB
  
  Solution: Enable multi-level spilling (not yet implemented)
  Reference: Phase B documentation
```

## 6. Testing & Validation

### Three-Tier Testing

1. **Assembly Validation** (fast)
   - Verify instruction mnemonics
   - Check function structure
   - Validate register allocation

2. **Execution Validation** (comprehensive)
   - Compile to ELF binary
   - Run on CPU/emulator
   - Compare against expected output
   - Check for crashes, memory errors

3. **Performance Profiling** (optional)
   - Measure execution time
   - Detect regressions
   - Profile CPU cache behavior

### Test Organization
```
tools/
├── test_legacy_scalar.cpp        # Legacy: 7 scalar tests
├── test_legacy_extended.cpp      # Legacy: 7 tensor tests
├── test_progressive_lowering.cpp # Progressive: 6 pipeline tests
├── test_execution.cpp            # NEW: Execution validation
├── test_utils.h                  # Assembly validation framework
└── execution_utils.h             # NEW: Execution framework
```

## 7. Integration Points

### IR Module
- `ir::Instruction` - Base for all instructions
- `ir::TensorOpInst` - Tensor operation specialization
- `ir::Function` - Function IR
- `ir::Type` - Type information (i64, f64, vectors)

### Compiler Pipeline
- **Frontend**: Parse → IR Construction
- **Optimization**: Dead code elimination, constant folding
- **Codegen**: ← (this module)
- **Linking**: Object file → Executable

### Runtime
- Standard library functions (printf, malloc, etc.)
- Calling convention support
- Exception handling (optional)

## 8. Performance Characteristics

### Compilation Time
```
Operation               | Time (ms)
────────────────────────┼──────────
Simple add              | 0.5 - 1.0
Complex scalar expr     | 2.0 - 5.0
8×8 tensor multiply     | 5.0 - 10.0
128×256 matrix multiply | 50.0 - 100.0
```

### Generated Code Quality
```
Metric                  | Scalar | Tensor
────────────────────────┼────────┼───────
Register spills         | 0-5%   | 0%
Code size ratio         | 1.2x   | 2.0x
Instructions/operation  | 3-5    | 50-100
Estimated IPC           | 1.5    | 3.0+ (with DMA)
```

### Memory Usage
```
8KB scratchpad (tensors)
 - InputA:     4KB (50% utilization)
 - InputB:     2KB (25% utilization)
 - Accumulator: 1KB (12% utilization)
 - Temporary:  1KB (13% utilization)
```

## References

- **Chaitin's Algorithm**: Register Allocation via Graph Coloring
- **Halide Paper**: A Language for Hardware-Aware Image Processing
- **TVM Project**: An Automated End-to-End Optimizing Compiler for Deep Learning
- **RISC-V ABI**: https://github.com/riscv/riscv-elf-psabi-doc
- **System V AMD64 ABI**: https://github.com/hjl-tools/x86-psABI
