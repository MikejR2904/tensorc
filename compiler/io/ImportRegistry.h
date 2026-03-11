#pragma once

#include "../ast/SymbolTable.h"
#include "../ast/Type.h"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>

struct ModuleExports
{
    std::string                             path;
    std::unordered_map<std::string, Symbol> symbols;

    bool define(Symbol sym)
    {
        auto [_, inserted] = symbols.emplace(sym.name, std::move(sym));
        return inserted;
    }

    const Symbol* lookup(const std::string& name) const
    {
        auto it = symbols.find(name);
        return (it != symbols.end()) ? &it->second : nullptr;
    }

    std::vector<std::string> exported_names() const
    {
        std::vector<std::string> out;
        out.reserve(symbols.size());
        for (auto& [n, _] : symbols) out.push_back(n);
        return out;
    }
};

class ImportRegistry
{
public:
    void add(const std::string& alias, std::shared_ptr<ModuleExports> mod)
    {
        modules_[alias] = std::move(mod);
    }

    bool has_module(const std::string& alias) const
    {
        return modules_.count(alias) > 0;
    }

    const Symbol* lookup(const std::string& alias,
                         const std::string& symbol_name) const
    {
        auto mit = modules_.find(alias);
        if (mit == modules_.end()) return nullptr;
        return mit->second->lookup(symbol_name);
    }

    std::optional<std::vector<std::string>>
    exported_names(const std::string& alias) const
    {
        auto mit = modules_.find(alias);
        if (mit == modules_.end()) return std::nullopt;
        return mit->second->exported_names();
    }

    // ── Built-in standard library stub ───────────────────────────────
    static ImportRegistry with_builtins()
    {
        ImportRegistry reg;
        Position p{0, 0};

        // ── std ───────────────────────────────────────────────────────
        {
            auto m = std::make_shared<ModuleExports>();
            m->path = "std";
            m->define(Symbol("print",    Type::fn({Type::infer()},  Type::void_()), IdentCtx::FuncDef, p));
            m->define(Symbol("println",  Type::fn({Type::infer()},  Type::void_()), IdentCtx::FuncDef, p));
            m->define(Symbol("read_line",Type::fn({},               Type::str_()), IdentCtx::FuncDef, p));
            m->define(Symbol("assert",   Type::fn({Type::bool_()},  Type::void_()), IdentCtx::FuncDef, p));
            m->define(Symbol("panic",    Type::fn({Type::str_()},   Type::void_()), IdentCtx::FuncDef, p));
            m->define(Symbol("exit",     Type::fn({Type::i32()},    Type::void_()), IdentCtx::FuncDef, p));
            reg.add("std", std::move(m));
        }

        // ── math ──────────────────────────────────────────────────────
        {
            auto m = std::make_shared<ModuleExports>();
            m->path = "math";
            for (const char* name : { "sqrt", "abs", "exp", "log",
                                      "sin", "cos", "tan", "floor", "ceil" })
                m->define(Symbol(name, Type::fn({Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
            m->define(Symbol("pow", Type::fn({Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
            m->define(Symbol("pi",  Type::f32(), IdentCtx::Def, p));
            m->define(Symbol("e",   Type::f32(), IdentCtx::Def, p));
            m->define(Symbol("inf", Type::f32(), IdentCtx::Def, p));
            reg.add("math", std::move(m));
        }

        // ── tensor ────────────────────────────────────────────────────
        {
            auto m = std::make_shared<ModuleExports>();
            m->path = "tensor";
            TypePtr shape_param = Type::array(Type::infer());
            TypePtr t_f32       = Type::tensor(Type::f32());
            TypePtr t_infer     = Type::tensor(Type::infer());
            m->define(Symbol("zeros",  Type::fn({shape_param}, t_f32),  IdentCtx::FuncDef, p));
            m->define(Symbol("ones",   Type::fn({shape_param}, t_f32),  IdentCtx::FuncDef, p));
            m->define(Symbol("rand",   Type::fn({shape_param}, t_f32),  IdentCtx::FuncDef, p));
            m->define(Symbol("randn",  Type::fn({shape_param}, t_f32),  IdentCtx::FuncDef, p));
            m->define(Symbol("full",   Type::fn({shape_param, Type::f32()}, t_f32), IdentCtx::FuncDef, p));
            m->define(Symbol("eye",    Type::fn({Type::infer()}, t_f32), IdentCtx::FuncDef, p));
            m->define(Symbol("arange", Type::fn({Type::infer(), Type::infer()}, t_f32), IdentCtx::FuncDef, p));

            m->define(Symbol("relu",      Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));
            m->define(Symbol("sigmoid",   Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));
            m->define(Symbol("tanh",      Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));
            m->define(Symbol("softmax",   Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));
            m->define(Symbol("leaky_relu",Type::fn({t_infer, Type::f32()}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("gelu",      Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));

            m->define(Symbol("sum",    Type::fn({t_infer}, Type::f32()),   IdentCtx::FuncDef, p));
            m->define(Symbol("mean",   Type::fn({t_infer}, Type::f32()),   IdentCtx::FuncDef, p));
            m->define(Symbol("max",    Type::fn({t_infer}, Type::f32()),   IdentCtx::FuncDef, p));
            m->define(Symbol("min",    Type::fn({t_infer}, Type::f32()),   IdentCtx::FuncDef, p));
            m->define(Symbol("argmax", Type::fn({t_infer}, Type::i64()),   IdentCtx::FuncDef, p));
            m->define(Symbol("argmin", Type::fn({t_infer}, Type::i64()),   IdentCtx::FuncDef, p));
            m->define(Symbol("norm",   Type::fn({t_infer}, Type::f32()),   IdentCtx::FuncDef, p));
            m->define(Symbol("std",    Type::fn({t_infer}, Type::f32()),   IdentCtx::FuncDef, p));

            m->define(Symbol("reshape",   Type::fn({t_infer, shape_param}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("transpose", Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));
            m->define(Symbol("flatten",   Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));
            m->define(Symbol("squeeze",   Type::fn({t_infer}, t_infer),   IdentCtx::FuncDef, p));
            m->define(Symbol("unsqueeze", Type::fn({t_infer, Type::infer()}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("cat",       Type::fn({Type::array(t_infer)}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("stack",     Type::fn({Type::array(t_infer)}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("split",     Type::fn({t_infer, Type::infer()}, Type::array(t_infer)), IdentCtx::FuncDef, p));
            m->define(Symbol("chunk",     Type::fn({t_infer, Type::infer()}, Type::array(t_infer)), IdentCtx::FuncDef, p));
            m->define(Symbol("slice",     Type::fn({t_infer, Type::infer(), Type::infer()}, t_infer), IdentCtx::FuncDef, p));

            m->define(Symbol("dot",       Type::fn({t_infer, t_infer}, Type::f32()),  IdentCtx::FuncDef, p));
            m->define(Symbol("matmul",    Type::fn({t_infer, t_infer}, t_infer),      IdentCtx::FuncDef, p));
            m->define(Symbol("cross",     Type::fn({t_infer, t_infer}, t_infer),      IdentCtx::FuncDef, p));
            m->define(Symbol("outer",     Type::fn({t_infer, t_infer}, t_infer),      IdentCtx::FuncDef, p));
            m->define(Symbol("inverse",   Type::fn({t_infer}, t_infer),               IdentCtx::FuncDef, p));
            m->define(Symbol("det",       Type::fn({t_infer}, Type::f32()),           IdentCtx::FuncDef, p));

            m->define(Symbol("exp",   Type::fn({t_infer}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("log",   Type::fn({t_infer}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("sqrt",  Type::fn({t_infer}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("abs",   Type::fn({t_infer}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("pow",   Type::fn({t_infer, Type::f32()}, t_infer), IdentCtx::FuncDef, p));
            m->define(Symbol("clamp", Type::fn({t_infer, Type::f32(), Type::f32()}, t_infer), IdentCtx::FuncDef, p));

            m->define(Symbol("backward", Type::fn({t_infer}, Type::void_()), IdentCtx::FuncDef, p));
            m->define(Symbol("no_grad",  Type::fn({t_infer}, t_infer),       IdentCtx::FuncDef, p));
            m->define(Symbol("detach",   Type::fn({t_infer}, t_infer),       IdentCtx::FuncDef, p));

            reg.add("tensor", std::move(m));
        }

        return reg;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<ModuleExports>> modules_;
};