/// codegen/lowering/MemoryLegalizer.cpp

#include "MemoryLegalizer.h"
#include <algorithm>
#include <cmath>

namespace codegen::lowering {

std::unique_ptr<LoopNest> MemoryLegalizer::legalize(const LoopNest& llir)
{
    auto legalized = std::make_unique<LoopNest>(llir);
    
    // Copy initial state
    legalized->buffers = llir.buffers;
    legalized->compute_blocks = llir.compute_blocks;
    legalized->dims = llir.dims;
    legalized->scratchpad_bytes = llir.scratchpad_bytes;
    
    // Phase B transformations
    handle_reduction_spilling(*legalized);
    inject_dma_operations(*legalized);
    inject_synchronization(*legalized);
    
    return legalized;
}

// ════════════════════════════════════════════════════════════════════════════
// Handle K > 64 Reduction Spilling
// ════════════════════════════════════════════════════════════════════════════

void MemoryLegalizer::handle_reduction_spilling(LoopNest& llir)
{
    // Detect reduction loops that exceed MAX_K
    const int64_t MAX_K = 64;
    
    // Find if there's a reduction loop
    auto reduction_it = std::find_if(llir.dims.begin(), llir.dims.end(),
        [](const LoopDim& d) { return d.is_reduction; });
    
    if (reduction_it == llir.dims.end()) {
        spilling_.requires_spilling = false;
        return;  // No reduction loops
    }
    
    // Check if reduction loop exceeds MAX_K
    int64_t reduction_range = reduction_it->limit - reduction_it->start;
    if (reduction_range <= MAX_K) {
        spilling_.requires_spilling = false;
        return;  // No spilling needed
    }
    
    // Need to inject outer spilling loop
    spilling_.requires_spilling = true;
    spilling_.outer_loop_iterations = (reduction_range + MAX_K - 1) / MAX_K;
    
    // Create outer k-spill loop (ko)
    // This loops over chunks of MAX_K elements
    auto ko_index = static_cast<int>(reduction_it - llir.dims.begin());
    LoopDim ko_loop("ko", 0, reduction_range, MAX_K, true, ko_index);
    
    // Modify the inner reduction loop to have MAX_K range
    reduction_it->start = 0;
    reduction_it->limit = MAX_K;
    reduction_it->nest_depth = ko_index + 1;
    
    // Insert the outer loop before the inner reduction
    llir.dims.insert(llir.dims.begin() + ko_index, ko_loop);
    
    // Update nesting depths for all subsequent dims
    for (size_t i = ko_index + 2; i < llir.dims.size(); ++i) {
        llir.dims[i].nest_depth++;
    }
    
    // Add memory copy operations for partial accumulator preservation
    // Before each outer iteration (except first), store C_partial
    ComputeBlock store_partial(ComputeKind::MemoryCopy, "store_partial_c");
    store_partial.operands.push_back({"C", {"i", "j"}});
    store_partial.operands.push_back({"C_preserve", {"ko"}});
    llir.compute_blocks.insert(llir.compute_blocks.begin(), store_partial);
    
    // After each outer iteration (except last), load C_partial
    ComputeBlock load_partial(ComputeKind::MemoryCopy, "load_partial_c");
    load_partial.operands.push_back({"C_preserve", {"ko"}});
    load_partial.operands.push_back({"C", {"i", "j"}});
    llir.compute_blocks.push_back(load_partial);
    
    // Add preservation buffer to memory
    // Size: M_tile × N_tile × outer_iterations
    auto preserve_buf = std::make_shared<MemoryBuffer>(
        "C_preserve", BufferRole::Temporary, 8, 
        8 * 8 * spilling_.outer_loop_iterations * 8, true
    );
    llir.add_buffer(preserve_buf);
}

// ════════════════════════════════════════════════════════════════════════════
// Inject DMA Operations
// ════════════════════════════════════════════════════════════════════════════

void MemoryLegalizer::inject_dma_operations(LoopNest& llir)
{
    // For each buffer used in compute blocks, ensure memory transfer operations
    // This is a simplified version; Phase C (Scheduler) will optimize these
    
    // Identify which buffers are main memory vs. scratchpad
    std::map<std::string, bool> is_scratchpad;
    for (const auto& buf : llir.buffers) {
        is_scratchpad[buf->name] = buf->is_scratchpad;
    }
    
    // Insert load operations before compute blocks for off-chip data
    std::vector<ComputeBlock> new_blocks;
    
    for (const auto& compute : llir.compute_blocks) {
        // Insert loads for input operands
        for (const auto& [buf_name, indices] : compute.operands) {
            // Skip if it's a pure output or already scratchpad
            if (compute.kind == ComputeKind::SystolicMatMul && buf_name == "C") continue;
            if (is_scratchpad[buf_name]) continue;
            
            // Insert DMA load before compute
            ComputeBlock dma_load(ComputeKind::MemoryCopy, "dma_load_" + buf_name);
            dma_load.operands.push_back({buf_name, indices});
            dma_load.operands.push_back({buf_name + "_local", indices});
            new_blocks.push_back(dma_load);
        }
        
        // Add the compute block itself
        new_blocks.push_back(compute);
        
        // Insert stores for output operands
        if (compute.kind == ComputeKind::SystolicMatMul || 
            compute.kind == ComputeKind::ElementWiseOp) {
            // Outputs go to C (accumulator), typically scratchpad
            // Will be stored to main memory after all iterations
        }
    }
    
    llir.compute_blocks = new_blocks;
}

// ════════════════════════════════════════════════════════════════════════════
// Inject Synchronization Points
// ════════════════════════════════════════════════════════════════════════════

void MemoryLegalizer::inject_synchronization(LoopNest& llir)
{
    // For Phase C scheduling, we need fences/scoreboards
    // This is a placeholder; actual synchronization depends on target hardware
    
    // If there are any DMA operations followed by compute, add a sync point
    bool has_dma = false;
    for (const auto& compute : llir.compute_blocks) {
        if (compute.kind == ComputeKind::MemoryCopy) {
            has_dma = true;
            break;
        }
    }
    
    if (!has_dma) return;
    
    // Insert synchronization barriers between memory and compute phases
    // This will be expanded in Phase C
    std::vector<ComputeBlock> synced_blocks;
    
    for (size_t i = 0; i < llir.compute_blocks.size(); ++i) {
        const auto& compute = llir.compute_blocks[i];
        synced_blocks.push_back(compute);
        
        // Add sync after DMA, before compute (if transitioning)
        if (compute.kind == ComputeKind::MemoryCopy) {
            if (i + 1 < llir.compute_blocks.size() &&
                llir.compute_blocks[i + 1].kind != ComputeKind::MemoryCopy) {
                ComputeBlock sync(ComputeKind::Synchronize, "dma_wait");
                synced_blocks.push_back(sync);
            }
        }
    }
    
    llir.compute_blocks = synced_blocks;
}

} // namespace codegen::lowering
