#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <stdexcept>
#include "ASTNode.h"
#include "Type.h"

struct Symbol {
    std::string name;
    TypePtr      type;
    IdentCtx    ctx;
    Position    pos;
    bool is_parallel_safe = true;
    bool requires_grad = false; 
    Symbol(std::string n, TypePtr t, IdentCtx c, Position p = {}) : name(std::move(n)), type(t), ctx(c), pos(p) {}
    bool is_infer() const { return !type || type->is_infer(); }
    const std::vector<Dim>& shape() const {
        static const std::vector<Dim> empty;
        return (type && type->kind == Type::Kind::Tensor) ? type->shape : empty;
    }
    const std::string& struct_name() const {
        static const std::string empty;
        return (type && type->kind == Type::Kind::Named) ? type->type_name : empty;
    }
    std::vector<TypePtr> param_types() const {
        if (type && type->kind == Type::Kind::Fn) return type->param_types();
        return {};
    }
    TypePtr ret_type() const {
        if (type && type->kind == Type::Kind::Fn) return type->ret_type();
        return Type::infer();
    }
};

class Scope {
public:
    Symbol* define(const Symbol& sym) {
        if (symbols.count(sym.name)) return nullptr; 
        auto [it, _] = symbols.emplace(sym.name, std::make_unique<Symbol>(sym));
        return it->second.get();
    }
    Symbol* resolve(const std::string& name) {
        auto it = symbols.find(name);
        if (it != symbols.end()) return it->second.get();
        return nullptr;
    }

private:
    std::unordered_map<std::string, std::unique_ptr<Symbol>> symbols;
};

class SymbolTable {
public:
    SymbolTable() { pushScope(); }
    void pushScope() { scopes.push_back(std::make_unique<Scope>()); }
    void popScope() {
        if (scopes.size() > 1) {
            scopes.pop_back();
        }
    }
    Symbol* define(const Symbol& sym) {
        Symbol* created = scopes.back()->define(sym);
        if (!created) {
            throw std::runtime_error("[" + std::to_string(sym.pos.line) + ":" + 
                std::to_string(sym.pos.column) + 
                "] Redefinition of identifier: " + sym.name);
        }
        return created;
    }

    Symbol* lookup(const std::string& name) {
        // Reverse iterate through the stack (closest scope first)
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            if (Symbol* sym = (*it)->resolve(name)) return sym;
        }
        return nullptr;
    }

    bool isGlobal(const std::string& name) {
        if (scopes.empty()) return false;
        return scopes.front()->resolve(name) != nullptr;
    }

    size_t depth() const { return scopes.size(); }

private:
    std::vector<std::unique_ptr<Scope>> scopes;
};