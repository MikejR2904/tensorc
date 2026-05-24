#pragma once

/// ParallelModuleHandler — lowers parallel::* calls to @parallel.* CallInsts.
/// Parallel execution and device management.
#include "../../io/module_handler.h"
#include "../IRBuilder.h"

namespace ir {

class ParallelModuleHandler : public io::ModuleHandler {
public:
    ir::Value* lower_call(ir::IRBuilder* builder, const std::string& func_name, const std::vector<ir::ValuePtr>& args, const TypePtr& ret_type) override;
    std::string module_name() const override { return "parallel"; }
    bool is_builtin() const override { return true; }
    bool supports_async() const override { return true; }
};

} // namespace ir
 