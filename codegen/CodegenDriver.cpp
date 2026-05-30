/// codegen/CodegenDriver.cpp
///
/// Unified code generation driver routing to legacy or progressive pipeline

#include "CodegenDriver.h"
#include "NewCodegenDriver.h"
#include "legacy/CodegenDriver.h"
#include <stdexcept>

namespace codegen {

CodegenDriver::CodegenDriver(const std::string& target)
    : target_(target)
{
}

bool CodegenDriver::lower_scalar_function(ir::Function* fn, const std::string& out_path)
{
    try {
        diagnostic_.clear();
        bool success = codegen::lower_function_to_asm(fn, out_path);
        if (!success) {
            diagnostic_ = "Legacy pipeline failed to lower scalar function: " + fn->name;
        }
        return success;
    } catch (const std::exception& e) {
        diagnostic_ = std::string("Scalar lowering exception: ") + e.what();
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
