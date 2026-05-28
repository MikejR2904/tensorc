#pragma once

/// codegen/lowering/ScratchpadAllocator.h
///
/// Phase B (Part 1): Static Scratchpad Address Allocation
///
/// For software-managed scratchpads (8 KB on Tensor-V, vs. hardware caches
/// on laptops), the compiler must act as the MMU, calculating absolute
/// physical byte offsets for every buffer allocation.
///
/// The allocator:
/// 1. Assigns buffers to scratchpad regions (offset within 8 KB)
/// 2. Tracks allocation state and detects conflicts
/// 3. Handles live-range analysis for reuse
/// 4. Reports over-allocation (triggers spilling in MemoryLegalizer)
///
/// Key insight from thesis: K > 64 spilling requires precise byte-level
/// offset tracking to avoid corrupting partial accumulator states.

#include "LoopNest.h"
#include <cstdint>
#include <map>
#include <vector>
#include <string>
#include <memory>

namespace codegen::lowering {

struct AllocationRecord {
    std::string buffer_name;
    int64_t offset;         // Byte offset within scratchpad
    int64_t size;           // Allocation size in bytes
    int64_t live_start;     // Loop nest index where first used
    int64_t live_end;       // Loop nest index where last used
    
    AllocationRecord(std::string n, int64_t off, int64_t sz, int64_t ls, int64_t le)
        : buffer_name(std::move(n)), offset(off), size(sz), live_start(ls), live_end(le) {}
};

enum class AllocationStrategy {
    Sequential,    // Allocate buffers back-to-back (simple, may waste space)
    GreedyReuse,   // Reuse space after live ranges end (compact)
    Optimal,       // ILP-based optimal packing (future)
};

// ════════════════════════════════════════════════════════════════════════════
// ScratchpadAllocator
// ════════════════════════════════════════════════════════════════════════════

class ScratchpadAllocator {
public:
    // Tensor-V scratchpad size from thesis
    static constexpr int64_t SCRATCHPAD_SIZE = 8 * 1024;  // 8 KB
    
    explicit ScratchpadAllocator(AllocationStrategy strat = AllocationStrategy::GreedyReuse)
        : strategy_(strat), total_allocated_(0) {}
    
    /// Analyze LoopNest and allocate all buffers to scratchpad offsets
    /// Returns allocation map: buffer_name → (offset, size)
    /// Throws std::runtime_error if allocation exceeds SCRATCHPAD_SIZE
    std::map<std::string, AllocationRecord> allocate(const LoopNest& llir);
    
    /// Query allocated offset for a buffer (after allocate() called)
    int64_t offset_of(const std::string& buffer_name) const;
    
    /// Total bytes allocated
    int64_t total_allocated() const { return total_allocated_; }
    
    /// Check if buffer is scratchpad-resident
    bool is_scratchpad_resident(const std::string& buffer_name) const;
    
private:
    AllocationStrategy strategy_;
    int64_t total_allocated_ = 0;
    std::map<std::string, AllocationRecord> allocations_;
    
    // Helper: sequential allocation strategy
    std::map<std::string, AllocationRecord> allocate_sequential(const LoopNest& llir);
    
    // Helper: greedy reuse strategy (respects live ranges)
    std::map<std::string, AllocationRecord> allocate_greedy_reuse(const LoopNest& llir);
    
    // Helper: analyze live ranges from loop nest
    void analyze_live_ranges(
        const LoopNest& llir,
        std::map<std::string, std::pair<int64_t, int64_t>>& ranges
    );
};

} // namespace codegen::lowering
