#include "nn_handler.h"
#include "../Instruction.h"

namespace ir {

ir::Value* NNModuleHandler::lower_call(ir::IRBuilder* builder, const std::string& func_name, const std::vector<ir::ValuePtr>& args, const TypePtr& ret_type)
{
    std::string mangled_name = "@nn." + func_name;
    Function* fn = nullptr;
    auto& global_funcs = builder->get_global_functions();
    auto it = global_funcs.find(mangled_name);
    if (it != global_funcs.end()) fn = it->second;
    else {
        std::vector<TypePtr> arg_types;
        for (const auto& a : args) arg_types.push_back(a ? a->type : Type::infer());
        fn = builder->get_module()->add_function(mangled_name, Type::fn(arg_types, ret_type), false);
        builder->get_global_functions()[mangled_name] = fn;
    }
    return builder->emit<CallInst>(ret_type->is_void() ? "" : builder->fresh(), ret_type, std::shared_ptr<Value>(fn, [](Value*){}), args, false);
}

} // namespace ir
