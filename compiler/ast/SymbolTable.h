#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <stdexcept>
#include "ASTNode.h"

struct Symbol {
    std::string name;
    TyKind      type;
    IdentCtx    ctx;
    Position    pos;
    std::vector<int> shape;
    std::vector<TyKind> param_types;
    std::optional<GenericParams> tensor_info;
    bool is_const = false;
    bool is_parallel_safe = true; 
    Symbol(std::string n, TyKind t, IdentCtx c, Position p, std::vector<int> s = {}, std::vector<TyKind> params = {})
        : name(std::move(n)), type(t), ctx(c), pos(p), shape(std::move(s)), param_types(std::move(params)) {}
};

class Scope {
public:
    // Returns nullptr if symbol already exists (shadowing/redefinition check)
    Symbol* define(const Symbol& sym) {
        if (symbols.find(sym.name) != symbols.end()) {
            return nullptr; 
        }
        auto [it, inserted] = symbols.emplace(sym.name, std::make_unique<Symbol>(sym));
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
    SymbolTable() {
        pushScope();
    }

    void pushScope() {
        scopes.push_back(std::make_unique<Scope>());
    }

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
            if (Symbol* sym = (*it)->resolve(name)) {
                return sym;
            }
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