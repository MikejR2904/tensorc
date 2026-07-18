/// codegen/CodegenDriver.cpp
///
/// Unified code generation driver routing to legacy or progressive pipeline

#include "CodegenDriver.h"
#include "NewCodegenDriver.h"
#include "bridge/TensorOpLowering.h"
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace codegen {

CodegenDriver::CodegenDriver(const std::string& target)
    : target_(target)
{
}

bool CodegenDriver::lower_scalar_function(ir::Function* fn, const std::string& out_path)
{
    try {
        diagnostic_.clear();
        ScalarCodegenPipeline pipeline(target_);
        if (!pipeline.valid()) {
            diagnostic_ = "Unknown target for scalar codegen: " + target_;
            return false;
        }
        std::ofstream out(out_path);
        if (!out) {
            diagnostic_ = "Failed to open output assembly path: " + out_path;
            return false;
        }
        bool success = pipeline.lower_function(*fn, out);
        if (!success) {
            diagnostic_ = "Scalar pipeline failed to lower function: " + fn->name;
        }
        return success;
    } catch (const std::exception& e) {
        diagnostic_ = std::string("Scalar lowering exception: ") + e.what();
        return false;
    }
}

bool CodegenDriver::lower_module_to_asm(ir::IRModule& mod, const std::string& out_path)
{
    try {
        diagnostic_.clear();

        bridge::TensorOpLoweringPass tensor_lowering(target_);
        tensor_lowering.run(mod);

        std::ofstream out(out_path);
        if (!out) {
            diagnostic_ = "Failed to open output assembly path: " + out_path;
            return false;
        }

        ScalarCodegenPipeline pipeline(target_);
        if (!pipeline.valid()) {
            diagnostic_ = "Unknown target for scalar codegen: " + target_;
            return false;
        }
        for (auto& fn : mod.functions) {
            if (!pipeline.lower_function(*fn, out)) {
                diagnostic_ = "Scalar pipeline failed to lower function: " + fn->name;
                return false;
            }
            out << "\n";
        }

        std::unordered_set<std::string> emitted_kernels;
        for (auto& fn : mod.functions) {
            for (auto& bb : fn->blocks) {
                for (auto& inst : bb->insts) {
                    auto* tensor_op = dynamic_cast<ir::TensorOpInst*>(inst.get());
                    if (!tensor_op || tensor_op->lowered_asm.empty()) continue;
                    if (!emitted_kernels.insert(tensor_op->name).second) continue;
                    out << tensor_op->lowered_asm << "\n";
                }
            }
        }

        return true;
    } catch (const std::exception& e) {
        diagnostic_ = std::string("Module lowering exception: ") + e.what();
        return false;
    }
}

void CodegenDriver::init_progressive_pipeline()
{
    if (!progressive_pipeline_) {
        progressive_pipeline_ = std::make_unique<ProgressiveLoweringPipeline>(target_);
    }
}

bool CodegenDriver::lower_tensor_operation(
    ir::TensorOpInst* hlir_op,
    const std::map<const void*, TensorShape>& shapes,
    std::ostream& out)
{
    try {
        diagnostic_.clear();
        init_progressive_pipeline();
        
        bool success = progressive_pipeline_->lower_tensor_op(hlir_op, shapes, out);
        if (!success) {
            diagnostic_ = progressive_pipeline_->last_diagnostic();
        }
        return success;
    } catch (const std::exception& e) {
        diagnostic_ = std::string("Tensor lowering exception: ") + e.what();
        return false;
    }
}

} // namespace codegen
