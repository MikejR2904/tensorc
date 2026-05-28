#pragma once

/// codegen/targets/X86TargetEmitter.h
///
/// Phase D: x86_64 Target Emitter
///
/// Generates x86_64 assembly for laptop CPUs using:
/// - Scalar loops for tiling
/// - SIMD intrinsics (AVX2/AVX-512 for vectorization)
/// - Standard x86_64 calling conventions
/// - cache-based memory hierarchy (no explicit scratchpad)

#include "TargetEmitter.h"

namespace codegen::targets {

class X86TargetEmitter : public TargetEmitter {
public:
    X86TargetEmitter() = default;
    
    const char* target_name() const override { return "x86_64"; }
    
    void emit(const LoopNest& mlir, std::ostream& out) override;
    
protected:
    void emit_loop_prologue(const LoopDim& dim, std::ostream& out) override;
    void emit_loop_epilogue(const LoopDim& dim, std::ostream& out) override;
    void emit_compute(const ComputeBlock& compute, std::ostream& out) override;
    void emit_memory_op(const ComputeBlock& memop, std::ostream& out) override;
    void emit_sync(const ComputeBlock& sync, std::ostream& out) override;
    
private:
    // x86_64 specific helpers
    void emit_matmul_kernel(const ComputeBlock& compute, std::ostream& out);
    void emit_elemwise_kernel(const ComputeBlock& compute, std::ostream& out);
    void emit_avx2_vadd(std::ostream& out);
    void emit_avx2_vmul(std::ostream& out);
    void emit_function_prologue(const std::string& name, std::ostream& out);
    void emit_function_epilogue(std::ostream& out);
    
    int loop_counter_ = 0;  // For generating unique loop labels
};

} // namespace codegen::targets
