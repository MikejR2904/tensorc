#pragma once

/// codegen/lowering/Tiler.h
///
/// Phase A: HLIR → LLIR Transformation via Tiling
///
/// Tiler converts high-level ir::TensorOpInst operations into explicit
/// nested loops over tiled memory buffers. Uses a fixed tile size of 8×8
/// to match the systolic array dimensions.
///
/// Rationale (from Halide paper):
///   - Compute: "What is being calculated" (the loop nest)
///   - Schedule: "How to calculate it" (tiling, vectorization)
/// This layer handles the Compute part for tensor operations.
///
/// Usage:
///   Tiler tiler;
///   auto llir = tiler.lower(hlir_matmul_inst, tensor_shapes);

#include "LoopNest.h"
#include "../../compiler/ir/Instruction.h"
#include "../../compiler/ir/Value.h"
#include <memory>
#include <map>
#include <cstdint>

namespace codegen::lowering {

struct TensorShape {
    std::vector<int64_t> dims;  // e.g., {128, 256} for 128×256 matrix
    int64_t element_bytes = 8;  // f64 = 8, f32 = 4
    
    int64_t total_elements() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    int64_t total_bytes() const {
        return total_elements() * element_bytes;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Tiler: Converts HLIR (TensorOpInst) → LLIR (LoopNest)
// ════════════════════════════════════════════════════════════════════════════

class Tiler {
public:
    // Fixed tile dimensions for systolic array
    static constexpr int64_t TILE_M = 8;
    static constexpr int64_t TILE_N = 8;
    static constexpr int64_t TILE_K = 8;
    
    // Max K reduction per inner loop (handles K > 64 case from thesis)
    static constexpr int64_t MAX_K_PER_BLOCK = 64;
    
    Tiler() = default;
    
    /// Lower a TensorOpInst (HLIR) to LoopNest (LLIR)
    /// shapes: map from operand Value* → shape information
    std::unique_ptr<LoopNest> lower(
        ir::TensorOpInst* hlir_op,
        const std::map<const void*, TensorShape>& shapes
    );
    
private:
    // Tile different operation types
    std::unique_ptr<LoopNest> tile_matmul(
        ir::TensorOpInst* op,
        int64_t M, int64_t N, int64_t K
    );
    
    std::unique_ptr<LoopNest> tile_element_wise(
        ir::TensorOpInst* op,
        const std::vector<int64_t>& shape
    );
    
    std::unique_ptr<LoopNest> tile_reduction(
        ir::TensorOpInst* op,
        const std::vector<int64_t>& shape
    );
    
    // Helper: Generate buffer references for a loop context
    std::vector<std::string> tile_indices(
        const std::vector<LoopDim>& dims,
        int depth  // How many loops to include
    );
};

} // namespace codegen::lowering
