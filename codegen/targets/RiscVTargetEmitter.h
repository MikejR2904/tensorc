#pragma once

/// codegen/targets/RiscVTargetEmitter.h
///
/// Phase D: RISC-V + Tensor-V Target Emitter
///
/// Generates code for in-order RISC-V core + custom decoupled accelerators:
/// - DMA controller (load/store command queues)
/// - Systolic matrix accelerator (8×8 tiled MxN×NxK)
/// - Software-managed scratchpad (8 KB)
///
/// Uses raw .insn directives for custom instructions beyond RV64I.
/// Implements decoupled execution model: issue DMA commands, compute,
/// synchronize via hardware scoreboard.

#include "TargetEmitter.h"

namespace codegen::targets {

class RiscVTargetEmitter : public TargetEmitter {
public:
    RiscVTargetEmitter() = default;
    
    const char* target_name() const override { return "riscv64"; }
    
    void emit(const LoopNest& mlir, std::ostream& out) override;
    
protected:
    void emit_loop_prologue(const LoopDim& dim, std::ostream& out) override;
    void emit_loop_epilogue(const LoopDim& dim, std::ostream& out) override;
    void emit_compute(const ComputeBlock& compute, std::ostream& out) override;
    void emit_memory_op(const ComputeBlock& memop, std::ostream& out) override;
    void emit_sync(const ComputeBlock& sync, std::ostream& out) override;
    
private:
    // RISC-V specific helpers
    void emit_systolic_matmul(const ComputeBlock& compute, std::ostream& out);
    void emit_elemwise_loop(const ComputeBlock& compute, std::ostream& out);
    void emit_dma_load_command(const ComputeBlock& memop, std::ostream& out);
    void emit_dma_store_command(const ComputeBlock& memop, std::ostream& out);
    void emit_scoreboard_wait(std::ostream& out);
    void emit_function_prologue(const std::string& name, std::ostream& out);
    void emit_function_epilogue(std::ostream& out);
    
    int loop_counter_ = 0;
    int dma_command_id_ = 0;
};

} // namespace codegen::targets
