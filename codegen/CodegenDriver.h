#pragma once

/// codegen/CodegenDriver.h
///
/// Unified Code Generation Driver
///
/// Routes IR to appropriate lowering pipeline:
/// - Scalar operations → Legacy pipeline (IR → MachineInstr → RegAlloc → Assembly)
/// - Tensor operations → Progressive pipeline (4-phase lowering)
///
/// Usage:
///   // Scalar operation
///   CodegenDriver driver;
///   driver.lower_scalar_function(fn, "output.s");
///
///   // Tensor operation
///   driver.lower_tensor_operation(matmul_op, shapes, std::cout);

#include "../compiler/ir/IRModule.h"
#include "../compiler/ir/Instruction.h"
#include "NewCodegenDriver.h"
#include "ScalarCodegenPipeline.h"
#include "lowering/Tiler.h"
#include <map>
#include <ostream>
#include <string>
#include <memory>

namespace codegen {

using lowering::TensorShape;

class CodegenDriver {
public:
    /// Construct with target architecture
    /// Supports: "x86_64", "riscv64"
    explicit CodegenDriver(const std::string& target = "riscv64");
    
    // ════════════════════════════════════════════════════════════════════════
    // Scalar Operations (Legacy Pipeline)
    // ════════════════════════════════════════════════════════════════════════
    
    /// Lower a scalar IR function to assembly file
    /// Returns true on success, false otherwise
    bool lower_scalar_function(ir::Function* fn, const std::string& out_path);

    /// Lower a full IR module. Tensor ops are first lowered into sidecar
    /// kernels through the bridge pass, then scalar/control-flow code is
    /// emitted through the legacy machine pipeline.
    bool lower_module_to_asm(ir::IRModule& mod, const std::string& out_path);
    
    // ════════════════════════════════════════════════════════════════════════
    // Tensor Operations (Progressive Lowering Pipeline)
    // ════════════════════════════════════════════════════════════════════════
    
    /// Lower a tensor operation through 4-phase pipeline
    /// shapes: map from operand Value* → shape information
    /// out: output assembly stream
    /// Returns true on success, false if operation not supported
    bool lower_tensor_operation(
        ir::TensorOpInst* hlir_op,
        const std::map<const void*, TensorShape>& shapes,
        std::ostream& out
    );
    
    /// Get diagnostic information from last lowering attempt
    const std::string& last_diagnostic() const { return diagnostic_; }
    
private:
    std::string target_;
    std::string diagnostic_;
    
    // Lazy-initialized pipeline components
    void init_progressive_pipeline();
    std::unique_ptr<ProgressiveLoweringPipeline> progressive_pipeline_;
};

} // namespace codegen
