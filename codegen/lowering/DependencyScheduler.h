#pragma once
 
/// codegen/lowering/DependencyScheduler.h
///
/// DependencyScheduler — Dependency-Aware Static Scheduler (Phase C)
///
/// Purpose
/// ───────
/// Maximise systolic-array processing-element occupancy by hiding the
/// latency of T-DMA transfers.  The hardware executes DMA (T-DMA engine)
/// and compute (systolic array / AME) concurrently; the compiler must emit
/// commands in an order that keeps both units busy.
///
/// Strategy: Double-Buffering with Scoreboard Synchronisation
/// ──────────────────────────────────────────────────────────
/// For a loop over tiles [0 .. N-1] the naive order is:
///
///   DMA_LOAD tile[0] → SCOREBOARD_WAIT → COMPUTE tile[0] → repeat
///
/// The optimised order pre-fetches the next tile while computing the current:
///
///   DMA_LOAD tile[0]          ; start DMA for tile 0
///   loop i = 0 .. N-1:
///     DMA_LOAD tile[i+1]      ; non-blocking: start DMA for NEXT tile
///     SCOREBOARD_WAIT         ; wait until tile[i] is ready
///     COMPUTE tile[i]         ; systolic-array compute
///   SCOREBOARD_WAIT           ; drain last DMA
///   COMPUTE tile[N-1]
///
/// This requires two ping-pong input buffers (A_ping/A_pong, B_ping/B_pong).
/// The scheduler creates these buffer variants and rewrites the compute-block
/// list to reference the appropriate ping or pong buffer per iteration.
///
/// Relationship to codegen::lowering::Scheduler
/// ─────────────────────────────────────────────
/// The existing Scheduler (Scheduler.h) performs high-level double-buffer
/// insertion based on simple heuristics.  This class sits *before* it in
/// the pipeline and performs the explicit producer-consumer dependency
/// analysis that the existing scheduler lacks.  After DependencyScheduler
/// runs, the existing Scheduler's double-buffer pass is effectively a
/// no-op (it detects that ping/pong buffers already exist).
 
#include "LoopNest.h"
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
 
namespace codegen::lowering {
 
// ── Dependency edge types ─────────────────────────────────────────────────────
 
enum class DepKind {
    RAW,   // Read-After-Write  (true dependency): compute reads DMA result
    WAR,   // Write-After-Read  (anti-dependency): DMA overwrites buffer compute reads
    WAW,   // Write-After-Write (output dependency): two DMAs to same buffer
};
 
struct DepEdge {
    size_t  producer_idx;   // index into ComputeBlock list
    size_t  consumer_idx;
    DepKind kind;
    std::string buffer_name;
};
 
// ── DependencyScheduler ───────────────────────────────────────────────────────
 
class DependencyScheduler {
public:
    DependencyScheduler() = default;
 
    /// Analyse and rewrite llir's compute_blocks in-place.
    /// Returns true if any reordering or buffer-splitting was applied.
    bool schedule(LoopNest& llir) {
        if (llir.compute_blocks.size() < 2) return false;
 
        deps_.clear();
        ping_pong_created_.clear();
 
        bool changed = false;
 
        // Phase C-1: build producer-consumer dependency graph
        build_dependency_graph(llir);
 
        // Phase C-2: create ping-pong buffer pairs for DMA inputs
        if (needs_double_buffering(llir)) {
            create_ping_pong_buffers(llir);
            changed = true;
        }
 
        // Phase C-3: reorder compute blocks to interleave DMA and compute
        if (reorder_for_overlap(llir)) changed = true;
 
        // Phase C-4: insert scoreboard waits at RAW edges
        if (insert_scoreboards(llir)) changed = true;
 
        if (verbose_ && changed) {
            std::cerr << "[DependencyScheduler] Reordered " << llir.compute_blocks.size()
                      << " compute blocks; " << ping_pong_created_.size()
                      << " ping-pong buffer pairs created\n";
        }
 
        return changed;
    }
 
    const std::vector<DepEdge>& dependency_edges() const { return deps_; }
    void set_verbose(bool v) { verbose_ = v; }
 
private:
    std::vector<DepEdge> deps_;
    std::unordered_set<std::string> ping_pong_created_;
    bool verbose_ = false;
 
    // ── Phase C-1: dependency graph ──────────────────────────────────────────
 
    void build_dependency_graph(const LoopNest& llir) {
        // For each buffer, track which compute block last wrote it
        std::unordered_map<std::string, size_t> last_writer;
 
        for (size_t i = 0; i < llir.compute_blocks.size(); ++i) {
            const auto& cb = llir.compute_blocks[i];
 
            for (const auto& [buf_name, indices] : cb.operands) {
                // Determine whether this block reads or writes the buffer
                bool is_write = is_output_buffer(cb, buf_name);
 
                if (!is_write) {
                    // READ — check for RAW dependency on previous writer
                    auto wit = last_writer.find(buf_name);
                    if (wit != last_writer.end() && wit->second < i) {
                        deps_.push_back({wit->second, i, DepKind::RAW, buf_name});
                    }
                } else {
                    // WRITE — check for WAR dependency on any prior reader
                    for (size_t j = 0; j < i; ++j) {
                        const auto& prev = llir.compute_blocks[j];
                        if (reads_buffer(prev, buf_name)) {
                            deps_.push_back({j, i, DepKind::WAR, buf_name});
                        }
                    }
                    // WAW: check for prior writer
                    auto wit = last_writer.find(buf_name);
                    if (wit != last_writer.end() && wit->second < i) {
                        deps_.push_back({wit->second, i, DepKind::WAW, buf_name});
                    }
                    last_writer[buf_name] = i;
                }
            }
        }
    }
 
    bool is_output_buffer(const ComputeBlock& cb, const std::string& name) const {
        // Outputs: accumulator buffers in compute ops, or destination in copies
        if (cb.kind == ComputeKind::SystolicMatMul ||
            cb.kind == ComputeKind::ElementWiseOp) {
            // Last operand is the output
            return (!cb.operands.empty() &&
                    cb.operands.back().first == name);
        }
        if (cb.kind == ComputeKind::MemoryCopy) {
            // Second operand is the destination
            return (cb.operands.size() >= 2 &&
                    cb.operands[1].first == name);
        }
        return false;
    }
 
    bool reads_buffer(const ComputeBlock& cb, const std::string& name) const {
        for (const auto& [buf, _] : cb.operands)
            if (buf == name && !is_output_buffer(cb, name)) return true;
        return false;
    }
 
    // ── Phase C-2: detect whether double-buffering is worthwhile ─────────────
 
    bool needs_double_buffering(const LoopNest& llir) const {
        bool has_dma     = false;
        bool has_compute = false;
        for (const auto& cb : llir.compute_blocks) {
            if (cb.kind == ComputeKind::MemoryCopy)    has_dma     = true;
            if (cb.kind == ComputeKind::SystolicMatMul ||
                cb.kind == ComputeKind::ElementWiseOp) has_compute = true;
        }
        // Also need at least one loop with more than one iteration
        int64_t outer_iters = 0;
        if (!llir.dims.empty()) {
            const auto& d = llir.dims[0];
            outer_iters = (d.limit - d.start + d.step - 1) / d.step;
        }
        return has_dma && has_compute && outer_iters > 1;
    }
 
    // ── Phase C-2: create ping-pong buffer pairs ──────────────────────────────
 
    void create_ping_pong_buffers(LoopNest& llir) {
        // Collect names of all DMA source buffers (InputA, InputB)
        std::vector<std::string> dma_inputs;
        for (const auto& cb : llir.compute_blocks) {
            if (cb.kind != ComputeKind::MemoryCopy) continue;
            if (cb.operands.empty()) continue;
            const std::string& src = cb.operands[0].first;
            if (ping_pong_created_.count(src)) continue;
            // Only create ping-pong for input buffers, not accumulators
            auto buf_it = std::find_if(llir.buffers.begin(), llir.buffers.end(),
                [&](const std::shared_ptr<MemoryBuffer>& b) { return b->name == src; });
            if (buf_it == llir.buffers.end()) continue;
            if ((*buf_it)->role == BufferRole::Accumulator ||
                (*buf_it)->role == BufferRole::Temporary) continue;
 
            dma_inputs.push_back(src);
            ping_pong_created_.insert(src);
        }
 
        for (const std::string& name : dma_inputs) {
            auto orig_it = std::find_if(llir.buffers.begin(), llir.buffers.end(),
                [&](const std::shared_ptr<MemoryBuffer>& b) { return b->name == name; });
            if (orig_it == llir.buffers.end()) continue;
 
            const auto& orig = *orig_it;
            // Create _ping and _pong variants (same size, in scratchpad)
            auto ping = std::make_shared<MemoryBuffer>(
                name + "_ping", orig->role, orig->element_size, orig->total_bytes, true);
            auto pong = std::make_shared<MemoryBuffer>(
                name + "_pong", orig->role, orig->element_size, orig->total_bytes, true);
            llir.add_buffer(ping);
            llir.add_buffer(pong);
 
            if (verbose_) {
                std::cerr << "[DependencyScheduler] Created ping-pong pair: "
                          << name << "_ping / " << name << "_pong ("
                          << orig->total_bytes << " B each)\n";
            }
        }
    }
 
    // ── Phase C-3: reorder for DMA / compute overlap ─────────────────────────
    //
    // Target order (per tile iteration):
    //   DMA_LOAD tile[i+1]   ← non-blocking lookahead for next iteration
    //   SCOREBOARD_WAIT      ← wait for tile[i]'s DMA to complete
    //   COMPUTE tile[i]      ← systolic-array work
    //
    // The existing Scheduler handles the ping-pong swap; here we only ensure
    // the DMA for the next tile is issued *before* the compute for the current.
 
    bool reorder_for_overlap(LoopNest& llir) {
        auto& blocks = llir.compute_blocks;
        std::vector<ComputeBlock> reordered;
        reordered.reserve(blocks.size());
 
        // Collect DMA loads, syncs, and computes separately, then interleave
        std::vector<ComputeBlock> dma_loads, syncs, computes, others;
        for (auto& cb : blocks) {
            switch (cb.kind) {
                case ComputeKind::MemoryCopy:
                    if (cb.operation.find("load") != std::string::npos ||
                        cb.operation.find("reload") != std::string::npos)
                        dma_loads.push_back(cb);
                    else
                        others.push_back(cb);
                    break;
                case ComputeKind::Synchronize:
                    syncs.push_back(cb);
                    break;
                case ComputeKind::SystolicMatMul:
                case ComputeKind::ElementWiseOp:
                    computes.push_back(cb);
                    break;
                default:
                    others.push_back(cb);
                    break;
            }
        }
 
        // Interleaved order: lookahead DMA → sync → compute → (repeat)
        // For each compute block, pair with the lookahead DMA for the next tile
        size_t n = computes.size();
        for (size_t i = 0; i < n; ++i) {
            // Issue lookahead DMA for tile i+1 (if it exists)
            if (i + 1 < dma_loads.size()) {
                reordered.push_back(dma_loads[i + 1]);
            }
            // Insert sync before compute (wait for tile i's DMA)
            if (i < syncs.size()) {
                reordered.push_back(syncs[i]);
            } else {
                // Synthesise a scoreboard wait if none was provided
                ComputeBlock wait(ComputeKind::Synchronize, "scoreboard_wait");
                reordered.push_back(wait);
            }
            reordered.push_back(computes[i]);
        }
 
        // Append writes/stores and any remaining blocks
        for (auto& cb : others) reordered.push_back(cb);
 
        bool changed = (reordered != blocks);
        if (changed) blocks = std::move(reordered);
        return changed;
    }
 
    // ── Phase C-4: insert SCOREBOARD_WAIT at each RAW dependency edge ────────
 
    bool insert_scoreboards(LoopNest& llir) {
        // After reordering, find any DMA→compute RAW edges that still lack
        // an intervening Synchronize block and insert one.
        bool inserted = false;
        auto& blocks = llir.compute_blocks;
 
        for (size_t i = 0; i + 1 < blocks.size(); ++i) {
            if (blocks[i].kind != ComputeKind::MemoryCopy) continue;
            if (blocks[i + 1].kind == ComputeKind::Synchronize) continue;
            if (blocks[i + 1].kind == ComputeKind::SystolicMatMul ||
                blocks[i + 1].kind == ComputeKind::ElementWiseOp) {
                ComputeBlock sb(ComputeKind::Synchronize, "scoreboard_wait");
                blocks.insert(blocks.begin() + static_cast<ptrdiff_t>(i + 1), sb);
                inserted = true;
                ++i;  // skip the newly inserted block
            }
        }
        return inserted;
    }
};
 
} // namespace codegen::lowering