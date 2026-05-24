#pragma once
 
/// MathModuleHandler — lowers  math::sin(x), math::pi, etc.
/// All math module functions are scalar (f32 → f32 or f32,f32 → f32).
/// They lower to CallInst targeting "@math.<name>" which the runtime
/// links to the C standard library (libm) or a built-in scalar FPU op.
/// Constants (math::pi, math::e, …) are lowered to ConstantFloat literals
/// so they are inlined and never produce a call at all.
 
#include "../../io/module_handler.h"
#include "../IRBuilder.h"
 
#include <optional>
#include <string>
 
namespace ir {
 
class MathModuleHandler : public io::ModuleHandler {
public:
    ir::Value* lower_call(
        ir::IRBuilder*                    builder,
        const std::string&               func_name,
        const std::vector<ir::ValuePtr>& args,
        const TypePtr&                   ret_type) override;
 
    std::string module_name() const override { return "math"; }
    bool is_builtin()         const override { return true;   }
 
private:
    /// If func_name is a known constant, return its value.
    std::optional<double> get_constant(const std::string& name) const;
 
    /// True if func_name is a known binary scalar function (f32,f32 → f32).
    bool is_binary(const std::string& name) const;
};
 
} // namespace ir
 