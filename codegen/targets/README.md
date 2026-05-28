# Target-Specific Code Emitters

**Location**: `codegen/targets/`

This module provides target abstraction for the progressive lowering pipeline. Instead of generating assembly directly, the lowering phases produce target-independent LLIR. This module then emits target-specific code.

## Design Pattern

### TargetEmitter (Abstract Base)

Base class defining the emission interface. Separates **what** to emit from **how** to emit it.

**Responsibilities**:
- Loop structure emission (for, while)
- Compute block emission (element-wise, matrix operations)
- Memory operations (loads, stores, DMA)
- Synchronization (scoreboards, fences)
- Architecture-specific instruction selection

**Virtual Methods**:
```cpp
// Compute blocks
virtual void emit_compute(const ComputeBlock& block) = 0;

// Memory operations
virtual void emit_dma_load(const DmaOp& op) = 0;
virtual void emit_dma_store(const DmaOp& op) = 0;

// Loops & control flow
virtual void emit_loop_begin(const LoopDim& dim) = 0;
virtual void emit_loop_end() = 0;

// Synchronization
virtual void emit_scoreboard_wait(uint32_t id) = 0;
virtual void emit_fence() = 0;
```

## Concrete Implementations

### X86TargetEmitter

**Target**: x86_64 with AVX2 SIMD

**Characteristics**:
- Scalar registers (RAX, RBX, RCX, ... RDX, R8-R15)
- Vector registers (YMM0-YMM15) for 256-bit operations
- Memory-to-register architecture (limited registers vs. RISC-V)

**Compute Emission**:
- 8×8 MatMul → unrolled scalar loops with AVX2 FMA instructions
- Element-wise → vectorized with VPADD, VMUL, etc.

**Memory Model**:
- Software-managed scratchpad mapped to stack-allocated buffers
- DMA emulated as memcpy calls
- Scoreboard waits map to memory barriers / dependency checks

**Assembly Format**:
```asm
.text
.global function_name
function_name:
    push rbp
    mov rsp, rbp
    sub rsp, <scratch_size>  # Allocate scratchpad
    
    # Emit DMA loads, compute, stores...
    # ...
    
    leave
    ret
```

### RiscVTargetEmitter

**Target**: RISC-V 64-bit with custom .insn directives

**Characteristics**:
- Universal register file (x0-x31 with ABI names)
- Floating-point registers (f0-f31)
- Vector registers (v0-v31)
- Load-store architecture (cleaner for DMA models)

**Custom Instructions** (via `.insn` directives):
- `AME_COMPUTE (0x7B, 0x0)`: Signal systolic array compute
- `DMA_LOAD (0x7B, 0x1)`: Initiate DMA load
- `DMA_STORE (0x7B, 0x2)`: Initiate DMA store
- `SCOREBOARD_WAIT (0x7B, 0x3)`: Wait for scoreboard event

**Compute Emission**:
- 8×8 MatMul → unrolled scalar loops with FMUL.D / FADD.D
- Alternative: Signal systolic array via AME_COMPUTE for specialized hardware

**Memory Model**:
- Hardware scratchpad (8 KB) allocated by ScratchpadAllocator
- DMA via custom instructions (overlaps with compute)
- Scoreboard events coordinate memory/compute pipeline

**Assembly Format**:
```asm
.text
.global function_name
function_name:
    # Emit compute with custom directives
    .insn r 0x7B, 0, 0, x0, x0, x0   # AME_COMPUTE
    .insn r 0x7B, 1, 0, x10, x5, x6  # DMA_LOAD rd, src, size
    
    # ...
    
    ret
```

## Integration with Progressive Lowering

**Flow**:
1. **Tiler** produces LLIR (loop nests, buffer references)
2. **ScratchpadAllocator** assigns buffer addresses
3. **MemoryLegalizer** injects DMA operations
4. **Scheduler** reorders for parallelism
5. **TargetEmitter** (this module) emits target assembly

**Factory Function**:
```cpp
std::unique_ptr<TargetEmitter> create_target_emitter(
    const std::string& target_name,
    std::ostream& output_stream
);
```

**Usage in NewCodegenDriver**:
```cpp
auto emitter = create_target_emitter("riscv64", asm_stream);
for (const auto& block : scheduled_blocks) {
    emitter->emit_compute(block);
}
```

## Adding New Targets

To support a new architecture (e.g., ARM64 with NEON):

1. **Create new header** (e.g., `Arm64TargetEmitter.h`):
   ```cpp
   #include "TargetEmitter.h"
   
   class Arm64TargetEmitter : public TargetEmitter {
       // Implement virtual methods
       void emit_compute(const ComputeBlock& block) override;
       void emit_dma_load(const DmaOp& op) override;
       // ... etc ...
   };
   ```

2. **Implement CPU-specific logic**:
   - NEON register file (v0-v31, 128-bit)
   - Instructions (FMUL, FADD, LDR, STR)
   - Memory model & caching

3. **Update factory function** in `TargetEmitter.cpp`:
   ```cpp
   if (target_name == "arm64")
       return std::make_unique<Arm64TargetEmitter>(output);
   ```

4. **Add tests** in `../tools/`:
   ```cpp
   // Test MatMul on ARM64
   auto emitter = create_target_emitter("arm64", asm_stream);
   // ...
   ```

## Key Design Principles

1. **Target Independence**: Lowering phases don't know about ISAs
2. **Extensibility**: New targets only require `TargetEmitter` subclass
3. **Clean Interface**: Emit operations, not instructions
4. **Flexibility**: Each target can choose compute/memory overlap strategy
5. **Diagnostics**: Error reporting via `last_diagnostic()` method

## See Also

- **../lowering/**: Phases that produce LLIR fed to emitters
- **../PROGRESSIVE_LOWERING.md**: Full architecture guide
- **../NewCodegenDriver.h/cpp**: Orchestrator that uses these emitters
