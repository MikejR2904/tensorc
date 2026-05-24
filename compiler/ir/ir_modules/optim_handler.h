#pragma once

/// OptimModuleHandler — lowers optim::* calls to @optim.* CallInsts.
/// Optimization algorithms: sgd, adam, adamw, etc.
#include "../../io/module_handler.h"
#include "../IRBuilder.h"

namespace ir {

class OptimModuleHandler : public io::ModuleHandler {
public:
    ir::Value* lower_call(ir::IRBuilder* builder, const std::string& func_name, const std::vector<ir::ValuePtr>& args, const TypePtr& ret_type) override;
    std::string module_name() const override { return "optim"; }
    bool is_builtin() const override { return true; }
    bool supports_async() const override { return true; }
};

} // namespace ir