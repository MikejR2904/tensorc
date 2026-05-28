#pragma once

/// codegen/lowering/MemoryLegalizer.h
///
/// Phase B (Part 2): Memory Address Legalization & Buffer Spilling
///
/// Handles the critical case from thesis: K > 64 reduction depth
/// exceeding the maximum block size for systolic processing.
///
/// Strategy:
/// 1. Detect when reduction loop (k-loop) exceeds MAX_K_PER_BLOCK (64)
/// 2. Inject outer spilling loop: processes in 64-element chunks
/// 3. Manage partial accumulator state across outer iterations
/// 4. Generate explicit memory operations for DMA transfers
///
/// Input: LLIR (LoopNest) from Tiler
/// Output: Legalized LLIR with spilling instructions inserted

#include "LoopNest.h"
#include "ScratchpadAllocator.h"
#include <memory>
#include <map>
#include <utility>

namespace codegen::lowering {

struct SpillingMetadata {
    bool requires_spilling;
    int64_t outer_loop_iterations;   // How many 64-element chunks?
    int64_t accumulator_preserve_offset;  // Where to store partial results
    int64_t temp_buffer_offset;          // Temporary memory for spilling
};

// ════════════════════════════════════════════════════════════════════════════
// MemoryLegalizer: Transform LLIR for hardware memory constraints
// ════════════════════════════════════════════════════════════════════════════

class MemoryLegalizer {
public:
    explicit MemoryLegalizer(
        ScratchpadAllocator allocator = ScratchpadAllocator()
    ) : allocator_(std::move(allocator)) {}
    
    /// Legalize LLIR: handle spilling, buffer conflicts, memory operations
    /// Input: Original LoopNest from Tiler
    /// Output: Legalized LoopNest with DMA operations and spilling handling
    std::unique_ptr<LoopNest> legalize(const LoopNest& hlir);
    
    /// Get metadata about spilling decisions
    const SpillingMetadata& spilling_info() const { return spilling_; }
    
private:
    ScratchpadAllocator allocator_;
    SpillingMetadata spilling_;
    
    // Helper: detect and transform K > 64 case
    void handle_reduction_spilling(LoopNest& llir);
    
    // Helper: inject memory copy operations for DMA
    void inject_dma_operations(LoopNest& llir);
    
    // Helper: add synchronization points between compute and memory
    void inject_synchronization(LoopNest& llir);
};

} // namespace codegen::lowering
