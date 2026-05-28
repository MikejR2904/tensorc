/// codegen/lowering/ScratchpadAllocator.cpp

#include "ScratchpadAllocator.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace codegen::lowering {

// ════════════════════════════════════════════════════════════════════════════
// Main allocation dispatcher
// ════════════════════════════════════════════════════════════════════════════

std::map<std::string, AllocationRecord> ScratchpadAllocator::allocate(const LoopNest& llir)
{
    allocations_.clear();
    total_allocated_ = 0;
    
    switch (strategy_) {
        case AllocationStrategy::Sequential:
            allocations_ = allocate_sequential(llir);
            break;
        case AllocationStrategy::GreedyReuse:
            allocations_ = allocate_greedy_reuse(llir);
            break;
        case AllocationStrategy::Optimal:
            // TODO: ILP-based optimal packing
            allocations_ = allocate_greedy_reuse(llir);
            break;
    }
    
    // Check total doesn't exceed capacity
    if (total_allocated_ > SCRATCHPAD_SIZE) {
        std::ostringstream oss;
        oss << "Scratchpad allocation (" << total_allocated_ << " bytes) "
            << "exceeds capacity (" << SCRATCHPAD_SIZE << " bytes). "
            << "Spilling required or tile size too large.";
        throw std::runtime_error(oss.str());
    }
    
    return allocations_;
}

int64_t ScratchpadAllocator::offset_of(const std::string& buffer_name) const
{
    auto it = allocations_.find(buffer_name);
    if (it == allocations_.end()) return -1;
    return it->second.offset;
}

bool ScratchpadAllocator::is_scratchpad_resident(const std::string& buffer_name) const
{
    return allocations_.find(buffer_name) != allocations_.end();
}

// ════════════════════════════════════════════════════════════════════════════
// Sequential Allocation
// ════════════════════════════════════════════════════════════════════════════

std::map<std::string, AllocationRecord> ScratchpadAllocator::allocate_sequential(const LoopNest& llir)
{
    std::map<std::string, AllocationRecord> result;
    int64_t offset = 0;
    
    for (const auto& buf : llir.buffers) {
        if (!buf->is_scratchpad) continue;
        
        // Allocate sequentially
        int64_t size = buf->total_bytes;
        result.emplace(buf->name, AllocationRecord(buf->name, offset, size, 0, llir.dims.size()));
        offset += size;
    }
    
    total_allocated_ = offset;
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Greedy Reuse Allocation (respects live ranges)
// ════════════════════════════════════════════════════════════════════════════

std::map<std::string, AllocationRecord> ScratchpadAllocator::allocate_greedy_reuse(const LoopNest& llir)
{
    // Step 1: Analyze live ranges for each buffer
    std::map<std::string, std::pair<int64_t, int64_t>> live_ranges;
    analyze_live_ranges(llir, live_ranges);
    
    // Step 2: Collect scratchpad-resident buffers
    std::vector<const MemoryBuffer*> scratchpad_bufs;
    for (const auto& buf : llir.buffers) {
        if (buf->is_scratchpad) {
            scratchpad_bufs.push_back(buf.get());
        }
    }
    
    // Step 3: Sort by size (descending) for better packing
    std::sort(scratchpad_bufs.begin(), scratchpad_bufs.end(),
        [](const MemoryBuffer* a, const MemoryBuffer* b) {
            return a->total_bytes > b->total_bytes;
        });
    
    // Step 4: Greedy allocation
    std::map<std::string, AllocationRecord> result;
    std::vector<std::pair<int64_t, int64_t>> free_ranges;  // (offset, size)
    free_ranges.push_back({0, SCRATCHPAD_SIZE});
    
    for (const auto* buf : scratchpad_bufs) {
        auto live_it = live_ranges.find(buf->name);
        int64_t live_start = live_it != live_ranges.end() ? live_it->second.first : 0;
        int64_t live_end = live_it != live_ranges.end() ? live_it->second.second : llir.dims.size();
        
        // Find first free range that fits
        int64_t allocated_offset = -1;
        for (size_t i = 0; i < free_ranges.size(); ++i) {
            auto& [off, sz] = free_ranges[i];
            if (sz >= buf->total_bytes) {
                allocated_offset = off;
                off += buf->total_bytes;
                sz -= buf->total_bytes;
                if (sz == 0) {
                    free_ranges.erase(free_ranges.begin() + i);
                }
                break;
            }
        }
        
        if (allocated_offset < 0) {
            std::ostringstream oss;
            oss << "Cannot allocate buffer '" << buf->name << "' ("
                << buf->total_bytes << " bytes)";
            throw std::runtime_error(oss.str());
        }
        
        result.emplace(buf->name, AllocationRecord(buf->name, allocated_offset, buf->total_bytes, live_start, live_end));
        total_allocated_ = std::max(total_allocated_, allocated_offset + buf->total_bytes);
    }
    
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Live Range Analysis
// ════════════════════════════════════════════════════════════════════════════

void ScratchpadAllocator::analyze_live_ranges(
    const LoopNest& llir,
    std::map<std::string, std::pair<int64_t, int64_t>>& ranges)
{
    // For each buffer, find first and last loop nesting level it's referenced
    
    // Initialize all buffers as not yet seen
    for (const auto& buf : llir.buffers) {
        if (buf->is_scratchpad) {
            ranges[buf->name] = {INT64_MAX, -1};
        }
    }
    
    // Scan through compute blocks (which reference buffers at certain loop depths)
    int64_t loop_index = 0;
    for (const auto& compute : llir.compute_blocks) {
        for (const auto& [buf_name, indices] : compute.operands) {
            auto it = ranges.find(buf_name);
            if (it != ranges.end()) {
                int64_t depth = indices.size();
                it->second.first = std::min(it->second.first, depth);
                it->second.second = std::max(it->second.second, depth);
            }
        }
        loop_index++;
    }
    
    // Clamp to valid ranges
    for (auto& [name, range] : ranges) {
        if (range.first == INT64_MAX) range.first = 0;
        if (range.second < 0) range.second = llir.dims.size();
    }
}

} // namespace codegen::lowering
