/// codegen/lowering/Scheduler.cpp

#include "Scheduler.h"
#include <algorithm>

namespace codegen::lowering {

std::unique_ptr<LoopNest> Scheduler::schedule(const LoopNest& legalized)
{
    auto scheduled = std::make_unique<LoopNest>(legalized);
    
    // Copy initial state
    scheduled->buffers = legalized.buffers;
    scheduled->dims = legalized.dims;
    scheduled->compute_blocks = legalized.compute_blocks;
    
    // Phase C transformations
    if (should_double_buffer(*scheduled)) {
        metadata_.uses_double_buffering = true;
        inject_ping_pong_buffers(*scheduled);
        reorder_for_overlap(*scheduled);
        inject_scoreboards(*scheduled);
    }
    
    return scheduled;
}

// ════════════════════════════════════════════════════════════════════════════
// Determine if double-buffering is beneficial
// ════════════════════════════════════════════════════════════════════════════

bool Scheduler::should_double_buffer(const LoopNest& llir)
{
    // Double-buffering is beneficial when:
    // 1. There are multiple loop iterations (pipelining across iterations)
    // 2. Both DMA and compute blocks present
    // 3. DMA latency > 0 (memory access overhead)
    
    bool has_memory_ops = false;
    bool has_compute_ops = false;
    int64_t iteration_count = 0;
    
    for (const auto& compute : llir.compute_blocks) {
        if (compute.kind == ComputeKind::MemoryCopy) has_memory_ops = true;
        if (compute.kind == ComputeKind::SystolicMatMul || 
            compute.kind == ComputeKind::ElementWiseOp) {
            has_compute_ops = true;
        }
    }
    
    // Count loop iterations (outer dimension)
    if (!llir.dims.empty()) {
        const auto& outer = llir.dims[0];
        iteration_count = (outer.limit - outer.start + outer.step - 1) / outer.step;
    }
    
    // Enable if we have both memory and compute, with multiple iterations
    return has_memory_ops && has_compute_ops && iteration_count > 1;
}

// ════════════════════════════════════════════════════════════════════════════
// Inject Ping/Pong Buffer Variants
// ════════════════════════════════════════════════════════════════════════════

void Scheduler::inject_ping_pong_buffers(LoopNest& scheduled)
{
    // For input buffers that are loaded via DMA, create Ping and Pong variants
    std::vector<std::string> dma_buffers;
    
    for (const auto& compute : scheduled.compute_blocks) {
        if (compute.kind == ComputeKind::MemoryCopy) {
            // First operand is the source (main memory)
            if (!compute.operands.empty()) {
                dma_buffers.push_back(compute.operands[0].first);
            }
        }
    }
    
    // Remove duplicates
    std::sort(dma_buffers.begin(), dma_buffers.end());
    dma_buffers.erase(std::unique(dma_buffers.begin(), dma_buffers.end()), dma_buffers.end());
    
    metadata_.ping_buffers = dma_buffers;
    
    // For each DMA buffer, create Ping and Pong variants
    for (const auto& buf_name : dma_buffers) {
        // Find original buffer
        auto orig_it = std::find_if(scheduled.buffers.begin(), scheduled.buffers.end(),
            [&](const std::shared_ptr<MemoryBuffer>& b) { return b->name == buf_name; });
        
        if (orig_it == scheduled.buffers.end()) continue;
        
        auto role = (*orig_it)->role;
        auto element_size = (*orig_it)->element_size;
        auto total_bytes = (*orig_it)->total_bytes;
        
        // Create Ping variant (in scratchpad)
        auto ping = std::make_shared<MemoryBuffer>(
            buf_name + "_ping", role, element_size, total_bytes, true
        );
        scheduled.add_buffer(ping);
        
        // Create Pong variant (in scratchpad)
        auto pong = std::make_shared<MemoryBuffer>(
            buf_name + "_pong", role, element_size, total_bytes, true
        );
        scheduled.add_buffer(pong);
        
        metadata_.pong_buffers.push_back(buf_name + "_pong");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Reorder Operations for Overlap
// ════════════════════════════════════════════════════════════════════════════

void Scheduler::reorder_for_overlap(LoopNest& scheduled)
{
    // Reorder compute blocks to pipeline:
    // DMA(Tile N+1) → Compute(Tile N) → Sync → [repeat]
    
    std::vector<ComputeBlock> reordered;
    size_t compute_idx = 0;
    
    // First iteration: just do compute (no DMA for N-1)
    // Find first compute block
    while (compute_idx < scheduled.compute_blocks.size() &&
           scheduled.compute_blocks[compute_idx].kind != ComputeKind::SystolicMatMul &&
           scheduled.compute_blocks[compute_idx].kind != ComputeKind::ElementWiseOp) {
        reordered.push_back(scheduled.compute_blocks[compute_idx]);
        compute_idx++;
    }
    
    if (compute_idx < scheduled.compute_blocks.size()) {
        reordered.push_back(scheduled.compute_blocks[compute_idx]);
        compute_idx++;
    }
    
    // Subsequent iterations: load next, compute current
    while (compute_idx < scheduled.compute_blocks.size()) {
        const auto& block = scheduled.compute_blocks[compute_idx];
        
        // Add DMA operation for next iteration (lookahead)
        if (block.kind == ComputeKind::MemoryCopy) {
            reordered.push_back(block);
        } else if (block.kind == ComputeKind::SystolicMatMul || 
                   block.kind == ComputeKind::ElementWiseOp) {
            // Before compute, check if there's a pending DMA
            // (This is simplified; real implementation would track dependencies)
            reordered.push_back(block);
        } else {
            reordered.push_back(block);
        }
        
        compute_idx++;
    }
    
    scheduled.compute_blocks = reordered;
}

// ════════════════════════════════════════════════════════════════════════════
// Insert Scoreboard/Fence Operations
// ════════════════════════════════════════════════════════════════════════════

void Scheduler::inject_scoreboards(LoopNest& scheduled)
{
    // Insert synchronization fences before operations that depend on DMA results
    
    std::vector<ComputeBlock> with_scoreboards;
    bool last_was_dma = false;
    
    for (const auto& compute : scheduled.compute_blocks) {
        // If last block was DMA and this is compute, add scoreboard
        if (last_was_dma && 
            (compute.kind == ComputeKind::SystolicMatMul || 
             compute.kind == ComputeKind::ElementWiseOp)) {
            ComputeBlock scoreboard(ComputeKind::Synchronize, "scoreboard_wait");
            with_scoreboards.push_back(scoreboard);
        }
        
        with_scoreboards.push_back(compute);
        last_was_dma = (compute.kind == ComputeKind::MemoryCopy);
    }
    
    scheduled.compute_blocks = with_scoreboards;
}

} // namespace codegen::lowering
