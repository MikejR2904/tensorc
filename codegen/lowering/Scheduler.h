#pragma once

/// codegen/lowering/Scheduler.h
///
/// Phase C: Asynchronous Double-Buffering & Task Scheduling
///
/// Prevents systolic stall when DMA transfers lag behind compute.
///
/// Pattern (from TVM/TPU papers):
///   DMA Load Tile N+1 (into "Pong" buffer)
///   Systolic Compute Tile N (from "Ping" buffer)
///   Swap Ping/Pong pointers
///   Fence/Scoreboard (hardware waits if DMA not done)
///
/// The scheduler:
/// 1. Analyzes memory latency and compute timing
/// 2. Reorders operations to maximize parallelism
/// 3. Injects double-buffer allocations (Ping/Pong)
/// 4. Inserts scoreboard/fence synchronization primitives
///
/// Input: Legalized LLIR (from MemoryLegalizer)
/// Output: MLIR with scheduled operations and double-buffering

#include "LoopNest.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace codegen::lowering {

struct ScheduleMetadata {
    bool uses_double_buffering = false;
    std::vector<std::string> ping_buffers;   // Buffers with Ping variant
    std::vector<std::string> pong_buffers;   // Buffers with Pong variant
    int64_t dma_latency_cycles = 50;         // Estimated DMA latency
    int64_t compute_cycles_per_tile = 64;    // Systolic tile latency
};

// ════════════════════════════════════════════════════════════════════════════
// Scheduler: Orchestrate double-buffering and task overlap
// ════════════════════════════════════════════════════════════════════════════

class Scheduler {
public:
    Scheduler() = default;
    
    /// Schedule a legalized LLIR for double-buffering and DMA overlap
    /// Input: LLIR with compute and memory blocks
    /// Output: MLIR with reordered operations and double-buffer management
    std::unique_ptr<LoopNest> schedule(const LoopNest& legalized);
    
    /// Get scheduling decisions
    const ScheduleMetadata& schedule_info() const { return metadata_; }
    
private:
    ScheduleMetadata metadata_;
    
    // Helper: analyze if double-buffering is beneficial
    bool should_double_buffer(const LoopNest& llir);
    
    // Helper: reorder operations for overlap
    void reorder_for_overlap(LoopNest& scheduled);
    
    // Helper: inject ping/pong buffer variants
    void inject_ping_pong_buffers(LoopNest& scheduled);
    
    // Helper: insert scoreboard/fence operations
    void inject_scoreboards(LoopNest& scheduled);
};

} // namespace codegen::lowering
