#pragma once

/// codegen/NewCodegenDriver.h
///
/// Progressive Lowering Pipeline Orchestrator
///
/// Coordinates the 4-phase transformation:
///   Phase A: HLIR → LLIR (Tiler)
///   Phase B: Legalization & Memory (ScratchpadAllocator, MemoryLegalizer)
///   Phase C: Scheduling & Double-Buffering (Scheduler)
///   Phase D: Target Emission (TargetEmitter)
///
/// Usage:
///   ProgressiveLoweringPipeline pipeline("riscv64");
///   pipeline.lower_tensor_op(hlir_matmul, shapes, output_stream);

#include "../compiler/ir/Instruction.h"
#include "lowering/LoopNest.h"
#include "lowering/Tiler.h"
#include "lowering/ScratchpadAllocator.h"
#include "lowering/MemoryLegalizer.h"
#include "lowering/Scheduler.h"
#include "targets/TargetEmitter.h"
#include <memory>
#include <map>
#include <ostream>
#include <string>

namespace codegen {

using lowering::TensorShape;
using lowering::LoopNest;
using targets::TargetEmitter;

// ════════════════════════════════════════════════════════════════════════════
// Progressive Lowering Pipeline
// ════════════════════════════════════════════════════════════════════════════

class ProgressiveLoweringPipeline {
public:
    /// Construct with target architecture
    /// Supports: "x86_64", "riscv64"
    explicit ProgressiveLoweringPipeline(const std::string& target);
    
    /// Lower a tensor operation through all phases and emit code
    /// shapes: map from operand Value* → shape information
    /// out: output assembly stream
    /// Returns true on success, false if operation not supported
    bool lower_tensor_op(
        ir::TensorOpInst* hlir_op,
        const std::map<const void*, TensorShape>& shapes,
        std::ostream& out
    );
    
    /// Get diagnostic information from last lowering
    const std::string& last_diagnostic() const { return diagnostic_; }
    
private:
    std::string target_;
    std::unique_ptr<TargetEmitter> emitter_;
    std::string diagnostic_;
    
    // Phase instances
    lowering::Tiler tiler_;
    lowering::ScratchpadAllocator allocator_;
    lowering::MemoryLegalizer legalizer_;
    lowering::Scheduler scheduler_;
};

} // namespace codegen
