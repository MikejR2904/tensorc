/// codegen/lowering/Tiler.cpp
///
/// Phase A Implementation: HLIR → LLIR Lowering with Tiling

#include "Tiler.h"
#include <cmath>
#include <algorithm>

namespace codegen::lowering {

std::unique_ptr<LoopNest> Tiler::lower(
    ir::TensorOpInst* hlir_op,
    const std::map<const void*, TensorShape>& shapes)
{
    if (!hlir_op) return nullptr;
    
    ir::TensorOpCode opcode = hlir_op->op;
    
    // Route to appropriate tiling strategy based on operation type
    switch (opcode) {
        // Matrix multiplication and related operations
        case ir::TensorOpCode::MatMul:
        case ir::TensorOpCode::Bmm:
        case ir::TensorOpCode::Dot: {
            // MatMul(A: m×k, B: k×n) → C: m×n
            // Expect exactly 2 arguments
            if (hlir_op->args.size() < 2) return nullptr;
            
            const void* a_ptr = hlir_op->args[0].get();
            const void* b_ptr = hlir_op->args[1].get();
            
            auto it_a = shapes.find(a_ptr);
            auto it_b = shapes.find(b_ptr);
            if (it_a == shapes.end() || it_b == shapes.end()) return nullptr;
            
            const auto& shape_a = it_a->second;
            const auto& shape_b = it_b->second;
            
            // Extract dimensions: A is m×k, B is k×n
            int64_t m = shape_a.dims[0];
            int64_t k = shape_a.dims[1];
            int64_t n = shape_b.dims[1];
            
            return tile_matmul(hlir_op, m, n, k);
        }
        
        // Element-wise operations
        case ir::TensorOpCode::ElemAdd:
        case ir::TensorOpCode::ElemMul:
        case ir::TensorOpCode::ElemSub:
        case ir::TensorOpCode::ElemDiv:
        case ir::TensorOpCode::Relu:
        case ir::TensorOpCode::Sigmoid: {
            if (hlir_op->args.empty()) return nullptr;
            
            auto it = shapes.find(hlir_op->args[0].get());
            if (it == shapes.end()) return nullptr;
            
            return tile_element_wise(hlir_op, it->second.dims);
        }
        
        // Reduction operations
        case ir::TensorOpCode::Sum:
        case ir::TensorOpCode::Mean:
        case ir::TensorOpCode::Max:
        case ir::TensorOpCode::Min: {
            if (hlir_op->args.empty()) return nullptr;
            
            auto it = shapes.find(hlir_op->args[0].get());
            if (it == shapes.end()) return nullptr;
            
            return tile_reduction(hlir_op, it->second.dims);
        }
        
        default:
            // Unimplemented operation
            return nullptr;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tile Matrix Multiplication: M×K × K×N → M×N with 8×8 tiles
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<LoopNest> Tiler::tile_matmul(
    ir::TensorOpInst* op,
    int64_t M, int64_t N, int64_t K)
{
    auto llir = std::make_unique<LoopNest>(op->name);
    
    // Create buffer references for A, B, C (output)
    auto buf_a = std::make_shared<MemoryBuffer>(
        "A", BufferRole::InputA, 8, M * K * 8, false
    );
    auto buf_b = std::make_shared<MemoryBuffer>(
        "B", BufferRole::InputB, 8, K * N * 8, false
    );
    auto buf_c = std::make_shared<MemoryBuffer>(
        "C", BufferRole::Accumulator, 8, TILE_M * TILE_N * 8, true  // Scratchpad tile
    );
    
    llir->add_buffer(buf_a);
    llir->add_buffer(buf_b);
    llir->add_buffer(buf_c);
    
    // Determine if spilling is needed (K > 64)
    bool needs_spilling = K > MAX_K_PER_BLOCK;
    llir->requires_spilling = needs_spilling;
    
    int depth = 0;
    
    // Outer loop: i-loop over tiles (M dimension)
    // i = 0, 8, 16, ..., M-1
    llir->add_loop_dim(LoopDim("i", 0, M, TILE_M, false, depth++));
    
    // Outer loop: j-loop over tiles (N dimension)
    // j = 0, 8, 16, ..., N-1
    llir->add_loop_dim(LoopDim("j", 0, N, TILE_N, false, depth++));
    
    if (needs_spilling) {
        // Outer K-loop for spilling: handles K > 64 case
        // ko = 0, 64, 128, ..., K-1
        llir->add_loop_dim(LoopDim("ko", 0, K, MAX_K_PER_BLOCK, true, depth++));
        
        // Inner K-loop for actual tiling
        // ki = 0, 8, 16, ..., min(64, K-ko)
        llir->add_loop_dim(LoopDim("ki", 0, MAX_K_PER_BLOCK, TILE_K, true, depth++));
    } else {
        // Simple case: k-loop over tiles directly
        // k = 0, 8, 16, ..., K-1
        llir->add_loop_dim(LoopDim("k", 0, K, TILE_K, true, depth++));
    }
    
    // At innermost level: systolic computation
    // Load tiles of A[i:i+8, k:k+8] and B[k:k+8, j:j+8]
    // Compute C[i:i+8, j:j+8] += A_tile × B_tile
    ComputeBlock compute(ComputeKind::SystolicMatMul, "matmul");
    compute.tile_m = TILE_M;
    compute.tile_n = TILE_N;
    compute.tile_k = TILE_K;
    
    // Operands: (buffer_name, [indices for this loop context])
    if (needs_spilling) {
        compute.operands.push_back({"A", {"i", "ko", "ki"}});
        compute.operands.push_back({"B", {"ko", "ki", "j"}});
        compute.operands.push_back({"C", {"i", "j", "ko"}});  // ko for partial accumulation
    } else {
        compute.operands.push_back({"A", {"i", "k"}});
        compute.operands.push_back({"B", {"k", "j"}});
        compute.operands.push_back({"C", {"i", "j"}});
    }
    
    llir->add_compute(compute);
    
    // Estimate scratchpad usage
    // C_tile[8×8 f64] = 512 bytes
    // A_tile[8×64 f64] = 4096 bytes
    // B_tile[64×8 f64] = 4096 bytes
    // Total ~= 8700 bytes (fits in 8 KB)
    llir->scratchpad_bytes = TILE_M * MAX_K_PER_BLOCK * 8  // A working set
                           + MAX_K_PER_BLOCK * TILE_N * 8  // B working set
                           + TILE_M * TILE_N * 8;          // C accumulator
    
    return llir;
}

// ════════════════════════════════════════════════════════════════════════════
// Tile Element-Wise Operations
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<LoopNest> Tiler::tile_element_wise(
    ir::TensorOpInst* op,
    const std::vector<int64_t>& shape)
{
    auto llir = std::make_unique<LoopNest>(op->name);
    
    // For element-wise ops, create buffers for each operand
    // Assume first operand is primary; tile accordingly
    
    if (shape.size() == 2) {
        // 2D tensor: tile to 8×8 blocks
        int64_t M = shape[0];
        int64_t N = shape[1];
        
        auto buf_in = std::make_shared<MemoryBuffer>(
            "input", BufferRole::InputA, 8, M * N * 8, false
        );
        auto buf_out = std::make_shared<MemoryBuffer>(
            "output", BufferRole::Accumulator, 8, TILE_M * TILE_N * 8, true
        );
        
        llir->add_buffer(buf_in);
        llir->add_buffer(buf_out);
        
        llir->add_loop_dim(LoopDim("i", 0, M, TILE_M, false, 0));
        llir->add_loop_dim(LoopDim("j", 0, N, TILE_N, false, 1));
        
        std::string op_name;
        switch (op->op) {
            case ir::TensorOpCode::Relu: op_name = "relu"; break;
            case ir::TensorOpCode::Sigmoid: op_name = "sigmoid"; break;
            case ir::TensorOpCode::ElemAdd: op_name = "add"; break;
            case ir::TensorOpCode::ElemMul: op_name = "mul"; break;
            default: op_name = "elemwise"; break;
        }
        
        ComputeBlock compute(ComputeKind::ElementWiseOp, op_name);
        compute.tile_m = TILE_M;
        compute.tile_n = TILE_N;
        compute.operands.push_back({"input", {"i", "j"}});
        compute.operands.push_back({"output", {"i", "j"}});
        
        llir->add_compute(compute);
        llir->scratchpad_bytes = TILE_M * TILE_N * 8 * 2;  // input + output tiles
        
    } else if (shape.size() == 1) {
        // 1D tensor: tile to 8-element blocks
        int64_t N = shape[0];
        
        auto buf_in = std::make_shared<MemoryBuffer>(
            "input", BufferRole::InputA, 8, N * 8, false
        );
        auto buf_out = std::make_shared<MemoryBuffer>(
            "output", BufferRole::Accumulator, 8, TILE_M * 8, true
        );
        
        llir->add_buffer(buf_in);
        llir->add_buffer(buf_out);
        
        llir->add_loop_dim(LoopDim("i", 0, N, TILE_M, false, 0));
        
        ComputeBlock compute(ComputeKind::ElementWiseOp, "elemwise_1d");
        compute.tile_m = TILE_M;
        compute.operands.push_back({"input", {"i"}});
        compute.operands.push_back({"output", {"i"}});
        
        llir->add_compute(compute);
        llir->scratchpad_bytes = TILE_M * 8 * 2;
    }
    
    return llir;
}

// ════════════════════════════════════════════════════════════════════════════
// Tile Reduction Operations
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<LoopNest> Tiler::tile_reduction(
    ir::TensorOpInst* op,
    const std::vector<int64_t>& shape)
{
    auto llir = std::make_unique<LoopNest>(op->name);
    
    // Reduction over entire tensor to scalar
    // Create a tiled reduction pattern
    
    if (shape.size() == 2) {
        int64_t M = shape[0];
        int64_t N = shape[1];
        
        auto buf_in = std::make_shared<MemoryBuffer>(
            "input", BufferRole::InputA, 8, M * N * 8, false
        );
        auto buf_acc = std::make_shared<MemoryBuffer>(
            "accumulator", BufferRole::Accumulator, 8, 8, true
        );
        
        llir->add_buffer(buf_in);
        llir->add_buffer(buf_acc);
        
        // Tile-based reduction: sum/max each 8×8 block, then reduce across blocks
        llir->add_loop_dim(LoopDim("i", 0, M, TILE_M, true, 0));  // reduction
        llir->add_loop_dim(LoopDim("j", 0, N, TILE_N, true, 1));  // reduction
        
        std::string op_name;
        switch (op->op) {
            case ir::TensorOpCode::Sum: op_name = "sum"; break;
            case ir::TensorOpCode::Max: op_name = "max"; break;
            case ir::TensorOpCode::Mean: op_name = "mean"; break;
            default: op_name = "reduce"; break;
        }
        
        ComputeBlock compute(ComputeKind::ElementWiseOp, op_name);
        compute.operands.push_back({"input", {"i", "j"}});
        compute.operands.push_back({"accumulator", {}});
        
        llir->add_compute(compute);
        llir->scratchpad_bytes = 64;  // Small accumulator
    }
    
    return llir;
}

} // namespace codegen::lowering
