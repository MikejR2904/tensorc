#pragma once

/// codegen/targets/TargetEmitter.h
///
/// Phase D: Abstract Target Machine Emitter
///
/// Decouples the MLIR (machine-agnostic scheduled operations) from
/// target-specific code generation. Each target (x86_64, ARM64, RISC-V)
/// inherits from TargetEmitter and implements:
///   - Loop emission
///   - Compute kernel dispatch
///   - Memory operation translation
///   - Register allocation
///
/// Follows the separation-of-concerns principle from TVM:
/// High-level operations → target-agnostic MLIR → target-specific emission

#include "../lowering/LoopNest.h"
#include <memory>
#include <string>
#include <ostream>

namespace codegen::targets {

using codegen::lowering::LoopNest;
using codegen::lowering::LoopDim;
using codegen::lowering::ComputeBlock;
using codegen::lowering::MemoryBuffer;

// ════════════════════════════════════════════════════════════════════════════
// Abstract Target Emitter
// ════════════════════════════════════════════════════════════════════════════

class TargetEmitter {
public:
    virtual ~TargetEmitter() = default;
    
    /// Get target name (x86_64, riscv64, etc.)
    virtual const char* target_name() const = 0;
    
    /// Emit MLIR (scheduled LoopNest) to target-specific assembly/code
    virtual void emit(
        const LoopNest& mlir,
        std::ostream& out
    ) = 0;
    
protected:
    // Helper methods for subclasses
    
    /// Emit loop prologue (set up counters, allocate registers)
    virtual void emit_loop_prologue(const LoopDim& dim, std::ostream& out) = 0;
    
    /// Emit loop epilogue and increment
    virtual void emit_loop_epilogue(const LoopDim& dim, std::ostream& out) = 0;
    
    /// Emit a compute operation (matmul, elemwise, etc.)
    virtual void emit_compute(const ComputeBlock& compute, std::ostream& out) = 0;
    
    /// Emit a memory operation (DMA load/store, copy)
    virtual void emit_memory_op(const ComputeBlock& memop, std::ostream& out) = 0;
    
    /// Emit synchronization (fence, barrier, scoreboard)
    virtual void emit_sync(const ComputeBlock& sync, std::ostream& out) = 0;
};

// ════════════════════════════════════════════════════════════════════════════
// Factory function
// ════════════════════════════════════════════════════════════════════════════

std::unique_ptr<TargetEmitter> create_target_emitter(const std::string& target);

} // namespace codegen::targets
