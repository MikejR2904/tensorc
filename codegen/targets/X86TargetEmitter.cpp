/// codegen/targets/X86TargetEmitter.cpp

#include "X86TargetEmitter.h"
#include <iomanip>
#include <functional>

namespace codegen::targets {

void X86TargetEmitter::emit(const LoopNest& mlir, std::ostream& out)
{
    out << "  .text\n";
    out << "  .global " << mlir.name << "\n";
    out << mlir.name << ":\n";
    
    emit_function_prologue(mlir.name, out);
    
    // Helper lambda for recursive loop emission
    std::function<void(size_t)> emit_loops = [&](size_t depth) -> void {
        if (depth >= mlir.dims.size()) {
            // At innermost level: emit compute blocks
            for (const auto& compute : mlir.compute_blocks) {
                switch (compute.kind) {
                    case codegen::lowering::ComputeKind::SystolicMatMul:
                    case codegen::lowering::ComputeKind::ElementWiseOp:
                        emit_compute(compute, out);
                        break;
                    case codegen::lowering::ComputeKind::MemoryCopy:
                        emit_memory_op(compute, out);
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

void X86TargetEmitter::emit_function_prologue(const std::string& name, std::ostream& out)
{
    out << "  push rbp\n";
    out << "  mov rbp, rsp\n";
    out << "  push rbx\n";
    out << "  push r12\n";
    out << "  push r13\n";
}

void X86TargetEmitter::emit_function_epilogue(std::ostream& out)
{
    out << "  pop r13\n";
    out << "  pop r12\n";
    out << "  pop rbx\n";
    out << "  pop rbp\n";
    out << "  ret\n";
}

// ════════════════════════════════════════════════════════════════════════════

void X86TargetEmitter::emit_loop_prologue(const LoopDim& dim, std::ostream& out)
{
    out << "  ; Loop: " << dim.name << " from " << dim.start 
        << " to " << dim.limit << " step " << dim.step << "\n";
    out << "  mov r" << (8 + loop_counter_) << ", " << dim.start << "\n";
    out << "L" << loop_counter_ << "_start:\n";
    out << "  cmp r" << (8 + loop_counter_) << ", " << dim.limit << "\n";
    out << "  jge L" << loop_counter_ << "_end\n";
    loop_counter_++;
}

void X86TargetEmitter::emit_loop_epilogue(const LoopDim& dim, std::ostream& out)
{
    loop_counter_--;
    out << "  add r" << (8 + loop_counter_) << ", " << dim.step << "\n";
    out << "  jmp L" << loop_counter_ << "_start\n";
    out << "L" << loop_counter_ << "_end:\n";
}

// ════════════════════════════════════════════════════════════════════════════

void X86TargetEmitter::emit_compute(const ComputeBlock& compute, std::ostream& out)
{
    out << "  ; Compute: " << compute.operation << "\n";
    
    if (compute.operation.find("matmul") != std::string::npos) {
        emit_matmul_kernel(compute, out);
    } else if (compute.operation.find("add") != std::string::npos ||
               compute.operation.find("mul") != std::string::npos) {
        emit_elemwise_kernel(compute, out);
    } else {
        // Generic element-wise
        out << "  ; TODO: emit " << compute.operation << "\n";
    }
}

void X86TargetEmitter::emit_memory_op(const ComputeBlock& memop, std::ostream& out)
{
    out << "  ; Memory: " << memop.operation << "\n";
    
    if (memop.operation.find("load") != std::string::npos) {
        out << "  ; DMA load operation\n";
        out << "  mov rax, [rdi]      ; Load from main memory\n";
        out << "  mov [rsi], rax      ; Store to scratchpad\n";
    } else if (memop.operation.find("store") != std::string::npos) {
        out << "  ; DMA store operation\n";
        out << "  mov rax, [rsi]      ; Load from scratchpad\n";
        out << "  mov [rdi], rax      ; Store to main memory\n";
    } else {
        out << "  ; Memory copy: " << memop.operation << "\n";
    }
}

void X86TargetEmitter::emit_sync(const ComputeBlock& sync, std::ostream& out)
{
    out << "  ; Synchronization: " << sync.operation << "\n";
    out << "  mfence              ; Memory fence\n";
}

// ════════════════════════════════════════════════════════════════════════════

void X86TargetEmitter::emit_matmul_kernel(const ComputeBlock& compute, std::ostream& out)
{
    // Simplified 8×8 matmul with AVX2
    out << "  ; 8x8 Matrix Multiply (AVX2)\n";
    out << "  call matmul_8x8_f64\n";
}

void X86TargetEmitter::emit_elemwise_kernel(const ComputeBlock& compute, std::ostream& out)
{
    if (compute.operation == "add") {
        emit_avx2_vadd(out);
    } else if (compute.operation == "mul") {
        emit_avx2_vmul(out);
    }
}

void X86TargetEmitter::emit_avx2_vadd(std::ostream& out)
{
    out << "  ; AVX2 vector add\n";
    out << "  vmovapd ymm0, [rdi]\n";
    out << "  vmovapd ymm1, [rsi]\n";
    out << "  vaddpd ymm2, ymm0, ymm1\n";
    out << "  vmovapd [rdx], ymm2\n";
}

void X86TargetEmitter::emit_avx2_vmul(std::ostream& out)
{
    out << "  ; AVX2 vector multiply\n";
    out << "  vmovapd ymm0, [rdi]\n";
    out << "  vmovapd ymm1, [rsi]\n";
    out << "  vmulpd ymm2, ymm0, ymm1\n";
    out << "  vmovapd [rdx], ymm2\n";
}

} // namespace codegen::targets
