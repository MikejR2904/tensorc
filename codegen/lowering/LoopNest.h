#pragma once

/// codegen/lowering/LoopNest.h
///
/// LLIR (Low-Level IR) Representation
///
/// Explicit nested loops over memory buffers, no high-level tensor ops.
/// Produced by Tiler (Phase A: HLIR → LLIR).
/// Input to MemoryLegalizer (Phase B).
///
/// Example: MatMul with 8×8 tiling
///   for i = 0, M, 8:
///     for j = 0, N, 8:
///       for k = 0, K, 64:        // Outer reduction (K>64 spilling)
///         for ki = 0, min(64,K-k), 8:  // Inner tile reduction
///           systolic_compute(...)

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace codegen::lowering {

// Forward declarations
struct MemoryBuffer;
struct LoopDim;
struct LoopNest;

// ════════════════════════════════════════════════════════════════════════════
// Memory Buffers (LLIR abstract memory references)
// ════════════════════════════════════════════════════════════════════════════

enum class BufferRole {
    InputA,         // Left operand (weights for systolic)
    InputB,         // Right operand (activations)
    Accumulator,    // Output / scratchpad partial results
    Temporary,      // Intermediate storage for spilling
};

struct MemoryBuffer {
    std::string name;
    BufferRole role;
    int64_t element_size;  // Bytes per element (8 for f64, 4 for f32)
    int64_t total_bytes;   // Total allocation size
    bool is_scratchpad;    // True if on software-managed memory
    
    MemoryBuffer(std::string n, BufferRole r, int64_t esize, int64_t tbytes, bool scratch = false)
        : name(std::move(n)), role(r), element_size(esize), total_bytes(tbytes), is_scratchpad(scratch) {}
};

// ════════════════════════════════════════════════════════════════════════════
// Loop Dimensions (LLIR induction variables)
// ════════════════════════════════════════════════════════════════════════════

struct LoopDim {
    std::string name;           // "i", "j", "k", "ki", etc.
    int64_t start;              // Loop start (usually 0)
    int64_t limit;              // Loop limit (exclusive)
    int64_t step;               // Increment per iteration
    bool is_reduction;          // True for k-loops (data dependency)
    int nest_depth;             // Nesting level (0=outermost)
    
    LoopDim(std::string n, int64_t s, int64_t l, int64_t st, bool red = false, int depth = 0)
        : name(std::move(n)), start(s), limit(l), step(st), is_reduction(red), nest_depth(depth) {}
};

// ════════════════════════════════════════════════════════════════════════════
// LoopNest: Explicit nested loop structure
// ════════════════════════════════════════════════════════════════════════════

enum class ComputeKind {
    SystolicMatMul,    // Activate systolic array (8×8 tile multiply-accumulate)
    ElementWiseOp,     // Scalar element-wise (add, mul, relu, etc.)
    MemoryCopy,        // DMA or loop-based buffer transfer
    Synchronize,       // Fence / barrier (for Phase C)
};

struct ComputeBlock {
    ComputeKind kind;
    std::string operation;     // "matmul", "add", "relu", "copy_a_to_scratch", etc.
    
    // Operands as (buffer_name, tile_indices)
    std::vector<std::pair<std::string, std::vector<std::string>>> operands;
    
    // For systolic/elementwise: tile dimensions (e.g., 8x8 for systolic)
    int64_t tile_m = 0, tile_n = 0, tile_k = 0;
    
    ComputeBlock(ComputeKind k, std::string op)
        : kind(k), operation(std::move(op)) {}
};

struct LoopNest {
    std::string name;                           // Function name
    std::vector<std::shared_ptr<MemoryBuffer>> buffers;  // All memory refs
    std::vector<LoopDim> dims;                  // Loop induction variables
    std::vector<ComputeBlock> compute_blocks;   // What to compute at innermost level
    
    // Metadata for Phase B/C
    int64_t scratchpad_bytes = 0;   // Total scratchpad allocation
    bool requires_spilling = false; // K > 64? Needs outer loop handling
    
    LoopNest(std::string n = "unknown") : name(std::move(n)) {}
    
    // Helpers
    std::shared_ptr<MemoryBuffer> get_buffer(const std::string& name) {
        for (auto& buf : buffers) if (buf->name == name) return buf;
        return nullptr;
    }
    
    void add_buffer(std::shared_ptr<MemoryBuffer> buf) { buffers.push_back(buf); }
    void add_loop_dim(const LoopDim& dim) { dims.push_back(dim); }
    void add_compute(const ComputeBlock& block) { compute_blocks.push_back(block); }
};

} // namespace codegen::lowering
