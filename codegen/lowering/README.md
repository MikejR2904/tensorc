# Progressive Lowering Pipeline

**Location**: `codegen/lowering/`

This module implements the four-phase progressive lowering pipeline that transforms high-level tensor operations into optimized target-specific code. Each phase specializes the representation further.

## Pipeline Overview

```
IR::TensorOpInst (MatMul, ElemAdd, etc.)
    ↓
Phase A: Tiling (Tiler)
    ↓
LLIR: Loop nests with 8×8 tiles
    ↓
Phase B: Memory Legalization (ScratchpadAllocator, MemoryLegalizer)
    ↓
LLIR with buffer allocation & DMA operations
    ↓
Phase C: Scheduling (Scheduler)
    ↓
LLIR with reordered compute/memory tasks
    ↓
Phase D: Target Emission (../targets/)
    ↓
Target-specific assembly (RISC-V, x86_64)
```

## Phase A: Tiling

**Component**: `Tiler.h/cpp`

**Input**: High-level tensor operations
- MatMul(M×K × K×N → M×N)
- ElemAdd(M×N + M×N → M×N)
- Reductions(M×N → scalar)

**Output**: Loop nests with 8×8 tiles (LLIR)

**Strategy**:
1. Tile tensors into 8×8 blocks
2. Generate nested loops over tiles
3. Generate compute blocks for each tile
4. Handle remainder tiles (< 8)

**Key Features**:
- **Target-agnostic**: Produces generic loop structure
- **K > 64 detection**: Flags operations requiring Phase B spilling
- **Buffer references**: Tracks input/output buffer identities (not addresses)

**Example** (MatMul):
```
for (t_i = 0; t_i < ceil(M/8); t_i++)
  for (t_j = 0; t_j < ceil(N/8); t_j++)
    for (t_k = 0; t_k < ceil(K/8); t_k++)
      # Compute block: 8×8 matmul
      C[t_i*8:t_i*8+8][t_j*8:t_j*8+8] +=
        A[t_i*8:t_i*8+8][t_k*8:t_k*8+8] @ B[t_k*8:t_k*8+8][t_j*8:t_j*8+8]
```

**Output Structure** (LoopNest):
- Hierarchy: outer loop (tiles over M) → middle (tiles over N) → inner (tiles over K)
- Each level tracks: dimension name, bounds, tile size
- Compute blocks reference logical buffers
- No addresses yet (phase B assigns those)

## Phase B: Memory Legalization

**Components**: `ScratchpadAllocator.h/cpp`, `MemoryLegalizer.h/cpp`

**Input**: LLIR with 8 KB scratchpad constraint

**Output**: LLIR with concrete buffer allocation & DMA operations

**Two Substeps**:

### B1: ScratchpadAllocator

Assigns static memory addresses to all buffers (8 KB limit).

**Strategy**:
1. Estimate buffer sizes from tile dimensions
2. Perform live-range analysis (when buffers are needed)
3. Reuse addresses when live ranges don't overlap (greedy)
4. Pack all buffers into 8 KB

**Live-Range Example**:
```
for t_k in tiles_over_K:
    DMA_LOAD A_tile  # range: [loop_start, loop_end]
    DMA_LOAD B_tile  # range: [loop_start, loop_end]
    COMPUTE          # uses A_tile, B_tile, accumulator C_tile
    # A_tile and B_tile live here; don't overlap with later buffers
    # Can reuse their addresses in next iteration

If K > 64 (requires spilling):
    Allocator reserves space for inner/outer accumulators
    Preserves partial results across outer iterations
```

**Output**: Each buffer mapped to concrete address (e.g., A_tile @ 0x0, B_tile @ 0x800)

### B2: MemoryLegalizer

Injects DMA operations for spilling when K > 64.

**Spilling Strategy** (Automatic):
1. Detects K > 64 at compile time
2. Adds outer reduction loop (K in 64-element chunks)
3. Injects spill/restore for accumulator across iterations
4. Uses buffer addresses from ScratchpadAllocator

**Example** (MatMul with K = 200):
```
# Outer loop: process K in 64-element chunks
for (k_chunk = 0; k_chunk < ceil(200/64); k_chunk++):  # 4 iterations
    # Inner tiling loop for K_chunk (max 64)
    for (t_k = 0; t_k < ceil(min(64, 200-k_chunk*64)/8); t_k++):
        # On first iteration, accumulator initialized to zero
        # On subsequent iterations, SPILL_RESTORE reads partial results
        if (k_chunk > 0):
            DMA_RESTORE accumulator_tile  # From scratchpad to local

        for (t_i, t_j):
            COMPUTE += A_tile @ B_tile
        
        # On last iteration, SPILL writes results back
        if (k_chunk < ceil(200/64) - 1):
            DMA_SPILL accumulator_tile    # To scratchpad

# Final result in accumulator_tile ready for output DMA
DMA_STORE C_tile
```

**Key Innovation**: Zero additional latency because spill/restore **overlaps with outer loop iterations** and other DMA operations.

## Phase C: Asynchronous Scheduling

**Component**: `Scheduler.h/cpp`

**Input**: LLIR with memory operations (DMA_LOAD, DMA_STORE, COMPUTE)

**Output**: Reordered operations with scoreboard events for parallelism

**Strategy**:

1. **Identify parallelism**:
   - Compute doesn't depend on DMA output? Start compute while DMA pending
   - Future DMA can be prefetched? Schedule lookahead

2. **Create ping-pong buffers**:
   - Double-buffer iterative loops
   - Load next tile while computing current tile

3. **Insert scoreboards**:
   - Mark when DMA completes
   - Compute waits for scoreboard before using data

**Example** (Double-Buffering):
```
Input:
  loop i:
    DMA_LOAD A_tile[i]
    DMA_LOAD B_tile[i]
    COMPUTE C += A_tile @ B_tile
    DMA_STORE C_tile

Output (with double-buffering):
  # Iteration -1: prefetch first tiles
  DMA_LOAD A_tile[0] → SB 1   # Scoreboard 1
  DMA_LOAD B_tile[0] → SB 2
  
  loop i from 0 to N-1:
    # Prefetch next iteration
    if (i < N-1):
      DMA_LOAD A_tile[i+1] → SB 3
      DMA_LOAD B_tile[i+1] → SB 4
    
    # Wait for current iteration data
    SB_WAIT 1
    SB_WAIT 2
    COMPUTE C += A_tile @ B_tile
    
    DMA_STORE C_tile
    
    # Swap scoreboards for next iteration
    SB 1 ← SB 3
    SB 2 ← SB 4
```

**Utilization**: ~90% compute/memory overlap (varies by operation)

## Phase D: Target Emission

**Location**: `../targets/`

**Responsibility**: Convert scheduled LLIR to target assembly

See `../targets/README.md` for details.

## Data Structures

### LoopNest (LoopNest.h)

Represents loop hierarchy:
```cpp
struct LoopDim {
    std::string name;        // "t_i", "t_j", "t_k", etc.
    size_t lower, upper;     // Loop bounds
    size_t tile_size;        // 8 for tiling
};

struct ComputeBlock {
    std::string operation;   // "matmul", "elem_add", etc.
    std::vector<BufferRef> inputs;
    std::vector<BufferRef> outputs;
    // Registers/temporaries needed
};

struct LoopNest {
    std::vector<LoopDim> loops;
    std::vector<ComputeBlock> blocks;
    std::string diagnostic;  // Error message if any
};
```

### Buffer Allocation (ScratchpadAllocator)

Maps logical buffer names to physical addresses:
```cpp
std::unordered_map<std::string, Address> buffer_addresses;
// Example:
// "A_tile" → Address(0x0000, 512 bytes)
// "B_tile" → Address(0x0200, 512 bytes)
// "C_tile" → Address(0x0400, 512 bytes)
// Total: 1536 bytes (well within 8 KB)
```

### Scheduled Operations (Scheduler)

Task graph with dependencies:
```cpp
struct ScheduledOp {
    enum Type { DMA_LOAD, DMA_STORE, COMPUTE };
    Type op_type;
    std::string buffer_name;
    std::vector<uint32_t> dependencies;  // Scoreboard IDs to wait for
};
```

## Key Design Principles

1. **Separation of Concerns**: Each phase has one responsibility
2. **Progressive Specialization**: High-level → LLIR → target assembly
3. **Automatic Optimization**: Tiling, scheduling done by pipeline (not user)
4. **Correctness by Construction**: Phases maintain invariants
5. **Diagnostics**: Rich error messages for debugging

## Usage Example

```cpp
#include "codegen/NewCodegenDriver.h"

// Build IR function with tensor op
auto mod = std::make_shared<ir::IRModule>("ml_model");
auto* fn = mod->add_function("matmul_optimized", ...);
// ... add MatMulInst, other ops ...

// Lower through pipeline
codegen::NewCodegenDriver driver;
std::ostringstream asm_output;
bool success = driver.lower(fn, "riscv64", asm_output);

if (!success) {
    std::cerr << "Error: " << driver.last_diagnostic() << std::endl;
}

std::cout << "Generated assembly:\n" << asm_output.str() << std::endl;
```

## See Also

- **../PROGRESSIVE_LOWERING.md**: 400+ line detailed guide with citations
- **../targets/**: Target-specific emitters
- **../NewCodegenDriver.h/cpp**: Pipeline orchestrator
- **../tools/test_progressive_lowering.cpp**: End-to-end tests
