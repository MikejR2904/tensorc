#pragma once
 
/// NNModuleHandler — lowers nn::linear, nn::conv2d, etc.
/// All nn functions lower to CallInst targeting "@nn.<name>" which the
/// runtime links to the neural-network kernel library.  No special IR is
/// needed at this stage; the backend dispatches to the correct kernel.
 
#include "../../io/module_handler.h"
#include "../IRBuilder.h"
 
namespace ir {
 
class NNModuleHandler : public io::ModuleHandler {
public:
    ir::Value* lower_call(ir::IRBuilder* builder, const std::string& func_name, const std::vector<ir::ValuePtr>& args, const TypePtr& ret_type) override;
    std::string module_name() const override { return "nn"; }
    bool is_builtin() const override { return true;  }
    bool supports_gpu() const override { return true;  }
};
 
} // namespace ir