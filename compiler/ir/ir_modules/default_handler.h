#pragma once
 
/// DefaultModuleHandler — generic IR lowering for user-defined .tcc modules.
/// User modules have no custom IR lowering; every exported function becomes a
/// plain CallInst targeting the mangled function name.  The function must
/// have been compiled and registered in the IRModule before this is called;
/// the linker resolves any remaining stubs.
///
/// Usage:
///   The compiler driver registers a DefaultModuleHandler for every user
///   module it compiles:
///     handler_registry.register_handler(
///         std::make_unique<ir::DefaultModuleHandler>("my_utils"));
///   After that, calls like  my_utils::normalize(x)  are lowered to
///     %r = call @my_utils::normalize(%x)
 
#include "../../io/module_handler.h"
#include "../IRBuilder.h"
 
namespace ir {
 
/// Generic handler for user-defined .tcc modules.
/// Emits a CallInst to "@<module>::<func>" for every call.
/// The function body is compiled separately and linked by the IR linker.
class DefaultModuleHandler : public io::ModuleHandler {
public:
    explicit DefaultModuleHandler(std::string module_name) : module_name_(std::move(module_name)) {}
    ir::Value* lower_call(ir::IRBuilder* builder, const std::string& func_name, const std::vector<ir::ValuePtr>& args, const TypePtr& ret_type) override;
    std::string module_name() const override { return module_name_; }
    bool is_builtin() const override { return false; }
 
private:
    std::string module_name_;
};
 
} // namespace ir