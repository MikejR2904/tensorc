#pragma once
 
#include "builtins.h"
 
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
 
namespace ir {
class IRBuilder;
struct Value;
using ValuePtr = std::shared_ptr<Value>;
}
 
namespace io {
 
// Forward declarations
class ModuleHandler;
using ModuleHandlerPtr = std::unique_ptr<ModuleHandler>;
 
/// Abstract base for module-specific IR lowering strategies.
/// Each module (tensor, std, math, nn, user-defined, etc.) registers a handler
/// that knows how to lower its function calls to IR instructions.
class ModuleHandler {
public:
    virtual ~ModuleHandler() = default;
 
    /// Attempt to lower a module function call to IR.
    /// @param builder  The IRBuilder instance (provides emit, scope access, etc.)
    /// @param func_name  Unqualified function name (e.g., "dot", not "tensor::dot")
    /// @param args  Pre-lowered IR values for arguments
    /// @param ret_type  Expected return type from semantic analysis
    /// @return IR Value* result, or nullptr if handler cannot lower this call
    /// If returns nullptr, IRBuilder will fall back to generic function call lowering.
    virtual ir::Value* lower_call(
        ir::IRBuilder* builder,
        const std::string& func_name,
        const std::vector<ir::ValuePtr>& args,
        const TypePtr& ret_type) = 0;
 
    /// Get the module name this handler manages (e.g., "tensor", "std", "math")
    virtual std::string module_name() const = 0;
 
    /// True if this is a built-in module, false if user-defined
    virtual bool is_builtin() const = 0;
 
    /// [OPTIONAL] Module capabilities/metadata for runtime context decisions
    virtual bool supports_gpu() const { return false; }
    virtual bool supports_async() const { return false; }
    virtual bool is_compile_time_only() const { return false; }
};
 
/// Registry for module handlers.
/// Stores handlers by module name and dispatches calls to the appropriate handler.
/// This decouples module-specific lowering logic from IRBuilder.
class ModuleHandlerRegistry {
public:
    ModuleHandlerRegistry() = default;
 
    /// Register a handler for a module.
    /// Takes ownership of the handler.
    void register_handler(ModuleHandlerPtr handler) {
        if (!handler) return;
        handlers_[handler->module_name()] = std::move(handler);
    }
 
    /// Get handler for a module, or nullptr if none registered.
    ModuleHandler* get_handler(const std::string& module_name) const {
        const std::string& canonical = resolve_alias(module_name);
        auto it = handlers_.find(canonical);
        return (it != handlers_.end()) ? it->second.get() : nullptr;
    }
 
    /// True if a handler exists for this module name or alias.
    bool has_handler(const std::string& module_name) const {
        return get_handler(module_name) != nullptr;
    }

    static ModuleHandlerRegistry with_builtins();
 
private:
    std::unordered_map<std::string, ModuleHandlerPtr> handlers_;
    std::unordered_map<std::string, std::string> aliases_; // alias → canonical
    
    /// Get all registered module names.
    std::vector<std::string> module_names() const {
        std::vector<std::string> names;
        names.reserve(handlers_.size());
        for (const auto& [name, _] : handlers_) {
            names.push_back(name);
        }
        return names;
    }
 
    /// Register an import alias: calls to `alias` are dispatched to the
    /// handler registered under `canonical`.
    /// e.g. register_alias("ts", "tensor") so ts::relu routes to TensorModuleHandler.
    void register_alias(const std::string& alias, const std::string& canonical) {
        if (alias != canonical) aliases_[alias] = canonical;
    }
 
    /// Resolve alias to canonical module name.
    const std::string& resolve_alias(const std::string& name) const {
        auto it = aliases_.find(name);
        return (it != aliases_.end()) ? it->second : name;
    }
 
    // (implementation consolidated above)
};
 
} // namespace io