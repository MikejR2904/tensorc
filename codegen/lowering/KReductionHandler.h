#pragma once
 
/// codegen/lowering/KReductionHandler.h
///
/// KReductionHandler — High-Pressure K>64 Reduction Handler (Phase B-3)
///
/// Problem: When the inner reduction dimension K > 64, the physical scratchpad
/// buffer configuration saturates. A naïve single-pass would require more
/// input-tile space than the 6 KB partition holds.
///
/// Solution: This pass intercepts the LoopNest and performs three transforms:
///
///   1. LOOP SPLIT — replaces the single k-loop with:
///        ko (outer): stride = MAX_K=64, loops over K in 64-element chunks
///        ki (inner): stride = TILE_K=8, tiles within each chunk
///
///   2. FLUSH + SWAP — at the end of each ko iteration injects:
///        SCOREBOARD_WAIT  (pipeline flush, ensures in-flight DMA completes)
///        accum_buffer_swap (ping-pong between C_ping and C_pong)
///
///   3. PARTIAL-SUM WRITEBACK — when the scratchpad accumulator would fill,
///        injects a DMA_STORE of the partial accumulator to main DRAM and a
///        DMA_LOAD to restore it at the start of the next ko iteration.
///        This creates zero extra latency when overlapped with compute.
///
/// The pass is idempotent: a LoopNest already containing a "ko" dimension
/// is left untouched.
 
#include "LoopNest.h"
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
 
namespace codegen::lowering {
 
class KReductionHandler {
public:
    static constexpr int64_t MAX_K  = 64;   // buffer-saturation threshold
    static constexpr int64_t TILE_K = 8;    // systolic tile K-depth
 
    KReductionHandler() = default;
 
    /// Transform llir in-place. Returns true when a split was performed.
    bool handle(LoopNest& llir) {
        // Idempotency guard
        for (const auto& dim : llir.dims)
            if (dim.name == "ko") return false;
 
        // Find the (first) reduction loop
        auto red_it = std::find_if(llir.dims.begin(), llir.dims.end(),
            [](const LoopDim& d) { return d.is_reduction; });
 
        if (red_it == llir.dims.end()) return false;
 
        int64_t K = red_it->limit - red_it->start;
        if (K <= MAX_K) return false;
 
        if (verbose_) {
            std::cerr << "[KReductionHandler] K=" << K
                      << " exceeds MAX_K=" << MAX_K
                      << " — splitting loop and injecting flush/swap/writeback\n";
        }
 
        split_k_loop(llir, red_it);
        inject_flush_and_swap(llir);
        inject_partial_sum_writeback(llir);
        llir.requires_spilling = true;
 
        return true;
    }
 
    void set_verbose(bool v) { verbose_ = v; }
 
private:
    bool verbose_ = false;
 
    // ── Step 1: Split k → ko (outer, stride MAX_K) + ki (inner, stride TILE_K)
    void split_k_loop(LoopNest& llir,
                      std::vector<LoopDim>::iterator red_it)
    {
        int64_t k_start = red_it->start;
        int64_t k_limit = red_it->limit;
        int     k_depth = red_it->nest_depth;
 
        LoopDim ko("ko", k_start, k_limit, MAX_K, /*is_reduction=*/true, k_depth);
        LoopDim ki("ki", 0,       MAX_K,   TILE_K, /*is_reduction=*/true, k_depth + 1);
 
        size_t pos = static_cast<size_t>(red_it - llir.dims.begin());
        llir.dims[pos] = ko;
        llir.dims.insert(llir.dims.begin() + static_cast<ptrdiff_t>(pos) + 1, ki);
 
        // Shift nesting depth of all dims that follow ki
        for (size_t i = pos + 2; i < llir.dims.size(); ++i)
            llir.dims[i].nest_depth++;
 
        // Rewrite index references: any operand index named "k" → "ko"
        // (the emitter computes the actual byte offset as ko + ki)
        for (auto& cb : llir.compute_blocks) {
            for (auto& [buf_name, indices] : cb.operands) {
                for (auto& idx : indices) {
                    if (idx == "k") idx = "ko";
                }
            }
        }
 
        if (verbose_) {
            int64_t ko_iters = (k_limit - k_start + MAX_K - 1) / MAX_K;
            std::cerr << "[KReductionHandler] ko: " << ko_iters
                      << " × " << MAX_K << " elements\n";
        }
    }
 
    // ── Step 2: Inject SCOREBOARD_WAIT + ping-pong swap after each ko iter ──
    void inject_flush_and_swap(LoopNest& llir) {
        // Pipeline flush — ensures all in-flight DMA ops for this ko chunk
        // have completed before the partial-sum writeback reads the accumulator.
        ComputeBlock flush(ComputeKind::Synchronize, "pipeline_flush");
        flush.operands.push_back({"scoreboard", {"ko"}});
 
        // Accumulator buffer swap — switches between ping and pong banks so
        // the next ko iteration writes into a clean accumulator tile while the
        // current partial sum is being DMA'd out.
        ComputeBlock swap(ComputeKind::Synchronize, "accum_buffer_swap");
        swap.operands.push_back({"C_ping", {"ko"}});
        swap.operands.push_back({"C_pong", {"ko"}});
 
        // Insert immediately after the last SystolicMatMul block
        auto last_mm = std::find_if(llir.compute_blocks.rbegin(),
                                    llir.compute_blocks.rend(),
            [](const ComputeBlock& cb) {
                return cb.kind == ComputeKind::SystolicMatMul;
            });
 
        if (last_mm != llir.compute_blocks.rend()) {
            // base() points one past the reverse iterator — i.e., after last_mm
            auto ins = last_mm.base();
            ins = llir.compute_blocks.insert(ins, flush);
            ++ins;
            llir.compute_blocks.insert(ins, swap);
        } else {
            llir.compute_blocks.push_back(flush);
            llir.compute_blocks.push_back(swap);
        }
    }
 
    // ── Step 3: Inject partial-sum writeback/reload around the outer ko loop ─
    void inject_partial_sum_writeback(LoopNest& llir) {
        // DMA_STORE: scratchpad accumulator → main DRAM
        // Emitted as: .insn r 0x7B, 0x2, 0x00, x0, a0, x0  ; DMA_STORE
        ComputeBlock writeback(ComputeKind::MemoryCopy, "partial_sum_writeback");
        writeback.operands.push_back({"C_partial", {"ko"}});   // src
        writeback.operands.push_back({"C_dram",    {"ko"}});   // dst
 
        // DMA_LOAD: main DRAM → scratchpad accumulator (start of next ko iter)
        // Emitted as: .insn r 0x7B, 0x1, 0x00, x0, a0, x0  ; DMA_LOAD
        ComputeBlock reload(ComputeKind::MemoryCopy, "partial_sum_reload");
        reload.operands.push_back({"C_dram",    {"ko"}});      // src
        reload.operands.push_back({"C_partial", {"ko"}});      // dst
 
        // Add the C_partial scratchpad buffer (8×8 FP32 = 256 bytes)
        bool has_partial = false;
        for (const auto& b : llir.buffers)
            if (b->name == "C_partial") { has_partial = true; break; }
 
        if (!has_partial) {
            auto buf = std::make_shared<MemoryBuffer>(
                "C_partial", BufferRole::Temporary,
                /*element_size=*/4, /*total_bytes=*/256, /*scratchpad=*/true);
            llir.add_buffer(buf);
        }
 
        // reload at front (executes before any compute in the ko iteration),
        // writeback at back (executes after flush+swap).
        llir.compute_blocks.insert(llir.compute_blocks.begin(), reload);
        llir.compute_blocks.push_back(writeback);
    }
};
 
} // namespace codegen::lowering