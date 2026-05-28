/// codegen/targets/RiscVTargetEmitter.cpp

#include "RiscVTargetEmitter.h"
#include <iomanip>
#include <functional>

namespace codegen::targets {

void RiscVTargetEmitter::emit(const LoopNest& mlir, std::ostream& out)
{
    out << "  .text\n";
    out << "  .global " << mlir.name << "\n";
    out << mlir.name << ":\n";
    
    emit_function_prologue(mlir.name, out);
    
    // Helper lambda for recursive loop emission
    std::function<void(size_t)> emit_loops = [&](size_t depth) -> void {
        if (depth >= mlir.dims.size()) {
            // Innermost: emit compute blocks with DMA pipelining
            for (size_t i = 0; i < mlir.compute_blocks.size(); ++i) {
                const auto& compute = mlir.compute_blocks[i];
                
                // Issue DMA for next iteration (lookahead)
                if (i + 1 < mlir.compute_blocks.size()) {
                    const auto& next = mlir.compute_blocks[i + 1];
                    if (next.kind == codegen::lowering::ComputeKind::MemoryCopy) {
                        emit_memory_op(next, out);
                    }
                }
                
                // Compute current iteration
                switch (compute.kind) {
                    case codegen::lowering::ComputeKind::SystolicMatMul:
                        emit_compute(compute, out);
                        break;
                    case codegen::lowering::ComputeKind::ElementWiseOp:
                        emit_compute(compute, out);
                        break;
                    case codegen::lowering::ComputeKind::MemoryCopy:
                        // Already handled in lookahead
                        break;
                    case codegen::lowering::ComputeKind::Synchronize:
                        emit_sync(compute, out);
                        break;
                }
            }
            return;
        }
        
        const auto& dim = mlir.dims[depth];
        emit_loop_prologue(dim, out);
        emit_loops(depth + 1);
        emit_loop_epilogue(dim, out);
    };
    
    emit_loops(0);
    
    emit_function_epilogue(out);
}

// ════════════════════════════════════════════════════════════════════════════

void RiscVTargetEmitter::emit_function_prologue(const std::string& name, std::ostream& out)
{
    out << "  addi sp, sp, -16\n";
    out << "  sd ra, 0(sp)\n";
    out << "  sd s0, 8(sp)\n";
}

void RiscVTargetEmitter::emit_function_epilogue(std::ostream& out)
{
    out << "  ld s0, 8(sp)\n";
    out << "  ld ra, 0(sp)\n";
    out << "  addi sp, sp, 16\n";
    out << "  ret\n";
}

// ════════════════════════════════════════════════════════════════════════════

void RiscVTargetEmitter::emit_loop_prologue(const LoopDim& dim, std::ostream& out)
{
    out << "  ; Loop: " << dim.name << " from " << dim.start 
        << " to " << dim.limit << " step " << dim.step << "\n";
    out << "  li t" << loop_counter_ << ", " << dim.start << "\n";
    out << "L" << loop_counter_ << "_start:\n";
    out << "  li t6, " << dim.limit << "\n";
    out << "  bge t" << loop_counter_ << ", t6, L" << loop_counter_ << "_end\n";
    loop_counter_++;
}

void RiscVTargetEmitter::emit_loop_epilogue(const LoopDim& dim, std::ostream& out)
{
    loop_counter_--;
    out << "  addi t" << loop_counter_ << ", t" << loop_counter_ << ", " << dim.step << "\n";
    out << "  j L" << loop_counter_ << "_start\n";
    out << "L" << loop_counter_ << "_end:\n";
}

// ════════════════════════════════════════════════════════════════════════════

void RiscVTargetEmitter::emit_compute(const ComputeBlock& compute, std::ostream& out)
{
    out << "  ; Compute: " << compute.operation << "\n";
    
    if (compute.operation.find("matmul") != std::string::npos) {
        emit_systolic_matmul(compute, out);
    } else {
        emit_elemwise_loop(compute, out);
    }
}

void RiscVTargetEmitter::emit_systolic_matmul(const ComputeBlock& compute, std::ostream& out)
{
    out << "  ; Systolic 8x8 MatMul (Tensor-V AME)\n";
    out << "  ; Setup tile dimensions\n";
    out << "  li a0, 8             ; m_tile\n";
    out << "  li a1, 8             ; n_tile\n";
    out << "  li a2, 8             ; k_tile\n";
    out << "  ; Load A, B pointers from loop context\n";
    out << "  ; (In real code, these come from buffer allocations)\n";
    out << "  ; Issue systolic command via custom instruction\n";
    out << "  .insn r 0x7B, 0x0, 0x00, x0, x0, x0  ; AME_COMPUTE\n";
    out << "  ; Scoreboard waits for completion\n";
    emit_scoreboard_wait(out);
}

void RiscVTargetEmitter::emit_elemwise_loop(const ComputeBlock& compute, std::ostream& out)
{
    out << "  ; Element-wise operation loop\n";
    if (compute.operation == "add") {
        out << "  ; Scalar add loop (RV64)\n";
        out << "  ld a0, 0(a0)\n";
        out << "  ld a1, 0(a1)\n";
        out << "  add a2, a0, a1\n";
        out << "  sd a2, 0(a2)\n";
    } else if (compute.operation == "mul") {
        out << "  ; Scalar mul loop (RV64)\n";
        out << "  ld a0, 0(a0)\n";
        out << "  ld a1, 0(a1)\n";
        out << "  mul a2, a0, a1\n";
        out << "  sd a2, 0(a2)\n";
    } else {
        out << "  ; TODO: " << compute.operation << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════

void RiscVTargetEmitter::emit_memory_op(const ComputeBlock& memop, std::ostream& out)
{
    if (memop.operation.find("load") != std::string::npos) {
        emit_dma_load_command(memop, out);
    } else if (memop.operation.find("store") != std::string::npos) {
        emit_dma_store_command(memop, out);
    } else {
        out << "  ; DMA operation: " << memop.operation << "\n";
    }
}

void RiscVTargetEmitter::emit_dma_load_command(const ComputeBlock& memop, std::ostream& out)
{
    out << "  ; DMA Load Command (Non-blocking)\n";
    out << "  ; Command ID: " << dma_command_id_ << "\n";
    
    // Simplified: load from a0 to a1, size in a2
    out << "  ; a0 = source (main memory)\n";
    out << "  ; a1 = destination (scratchpad)\n";
    out << "  ; a2 = size (bytes)\n";
    
    // Issue DMA load via custom instruction
    out << "  li t0, " << dma_command_id_ << "      ; DMA command ID\n";
    out << "  .insn r 0x7B, 0x1, 0x00, x0, a0, x0  ; DMA_LOAD\n";
    
    dma_command_id_++;
}

void RiscVTargetEmitter::emit_dma_store_command(const ComputeBlock& memop, std::ostream& out)
{
    out << "  ; DMA Store Command (Non-blocking)\n";
    out << "  ; a0 = source (scratchpad)\n";
    out << "  ; a1 = destination (main memory)\n";
    out << "  ; a2 = size (bytes)\n";
    
    out << "  li t0, " << dma_command_id_ << "      ; DMA command ID\n";
    out << "  .insn r 0x7B, 0x2, 0x00, x0, a0, x0  ; DMA_STORE\n";
    
    dma_command_id_++;
}

void RiscVTargetEmitter::emit_sync(const ComputeBlock& sync, std::ostream& out)
{
    emit_scoreboard_wait(out);
}

void RiscVTargetEmitter::emit_scoreboard_wait(std::ostream& out)
{
    out << "  ; Scoreboard synchronization\n";
    out << "  .insn r 0x7B, 0x3, 0x00, x0, x0, x0  ; SCOREBOARD_WAIT\n";
}

} // namespace codegen::targets
