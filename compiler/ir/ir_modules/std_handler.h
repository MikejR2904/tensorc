#pragma once

#include "../../io/module_handler.h"
#include "../IRBuilder.h"

namespace ir {

/// Handler for the built-in "std" module.
/// Lowers standard library functions to CallInst IR instructions.
/// Most std functions are externally defined and called generically.
class StdModuleHandler : public io::ModuleHandler {
public:
    ir::Value* lower_call(ir::IRBuilder* builder, const std::string& func_name, const std::vector<ir::ValuePtr>& args, const TypePtr& ret_type) override
    {
        // Look for a built-in function named "@std.function_name"
        Function* std_fn = nullptr;
        std::string mangled_name = "@std." + func_name;

        // Try to find it in global functions
        auto& global_funcs = builder->get_global_functions();
        auto it = global_funcs.find(mangled_name);
        if (it != global_funcs.end()) {
            std_fn = it->second;
        } else {
            // Auto-create function stub (will be resolved by runtime/linker)
            std::vector<TypePtr> arg_types;
            arg_types.reserve(args.size());
            for (const auto& v : args) {
                arg_types.push_back(v->type);
            }
            ir::IRModule* mod = builder->get_module();
            std_fn = mod->add_function(mangled_name, Type::fn(arg_types, ret_type), false);
            builder->get_global_functions()[mangled_name] = std_fn;
        }

        // Emit generic call instruction
        return builder->emit<CallInst>(
            ret_type->is_void() ? "" : builder->fresh(), ret_type, std::shared_ptr<Value>(std_fn, [](Value*) {}), args, false
        );
    }

    std::string module_name() const override { return "std"; }
    bool is_builtin() const override { return true; }
};

} // namespace ir
