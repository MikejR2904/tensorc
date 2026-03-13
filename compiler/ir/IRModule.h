#pragma once
 
#include "Instruction.h"
#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
 
namespace ir {
 
// ─── BasicBlock ──────────────────────────────────────────────────────────────
 
/// A maximal straight-line sequence of instructions ending in a terminator
/// (BranchInst, CondBranchInst, or ReturnInst).
struct BasicBlock
{
    std::string            label;       ///< e.g. "entry", "if.true", "loop.body"
    std::vector<InstPtr>   insts;       ///< ordered instruction list
    Function*              parent = nullptr;
 
    /// Predecessor blocks (populated by CFG-building pass).
    std::vector<BasicBlock*> preds;
    /// Successor blocks (derived from terminator).
    std::vector<BasicBlock*> succs;
 
    explicit BasicBlock(std::string label) : label(std::move(label)) {}
 
    // ── Instruction emission ─────────────────────────────────────────────────
 
    template<typename T, typename... Args>
    T* emit(Args&&... args)
    {
        auto inst = std::make_shared<T>(std::forward<Args>(args)...);
        inst->parent = this;
        insts.push_back(inst);
        return static_cast<T*>(insts.back().get());
    }
 
    /// True if the last instruction is a terminator.
    bool is_terminated() const
    {
        if (insts.empty()) return false;
        auto* last = insts.back().get();
        return dynamic_cast<BranchInst*>(last)     ||
               dynamic_cast<CondBranchInst*>(last) ||
               dynamic_cast<ReturnInst*>(last);
    }
 
    Instruction* terminator()
    {
        return is_terminated() ? insts.back().get() : nullptr;
    }
};
 
// ─── Function ────────────────────────────────────────────────────────────────
 
/// Represents a compiled fn or async fn definition.
struct Function : Value
{
    std::vector<std::shared_ptr<Argument>> params;
    std::vector<std::shared_ptr<BasicBlock>> blocks;
 
    bool is_async    = false;
    bool is_exported = false;   ///< true for top-level exported symbols
 
    IRModule* parent_module = nullptr;
 
    Function(std::string name, TypePtr fn_type)
        : Value(std::move(name), std::move(fn_type)) {}
 
    // ── Block management ─────────────────────────────────────────────────────
 
    BasicBlock* entry()
    {
        assert(!blocks.empty() && "Function has no blocks");
        return blocks.front().get();
    }
 
    BasicBlock* add_block(std::string label)
    {
        auto bb = std::make_shared<BasicBlock>(std::move(label));
        bb->parent = this;
        blocks.push_back(std::move(bb));
        return blocks.back().get();
    }
 
    BasicBlock* create_entry()
    {
        assert(blocks.empty() && "entry already exists");
        return add_block("entry");
    }
 
    // ── Parameter management ─────────────────────────────────────────────────
 
    Argument* add_param(std::string name, TypePtr type)
    {
        auto arg = std::make_shared<Argument>(
            std::move(name), std::move(type), params.size());
        params.push_back(std::move(arg));
        return params.back().get();
    }
 
    // ── Value lookup (by name, for IR builder use) ───────────────────────────
 
    Value* find_value(const std::string& name) const
    {
        for (auto& p : params)
            if (p->name == name) return p.get();
        for (auto& bb : blocks)
            for (auto& inst : bb->insts)
                if (inst->name == name) return inst.get();
        return nullptr;
    }
};
 
// ─── IRModule ────────────────────────────────────────────────────────────────
 
/// Top-level container for a compiled .tcc source file.
/// One IRModule per compilation unit; linked together by the import resolver.
struct IRModule
{
    std::string source_path;   ///< absolute path of the originating .tcc file
 
    std::vector<std::shared_ptr<Function>>    functions;
    std::vector<std::shared_ptr<Value>>       globals;    ///< global let bindings
 
    /// Modules imported by this one (resolved by ImportResolver).
    std::vector<IRModule*> imports;
 
    explicit IRModule(std::string path) : source_path(std::move(path)) {}
 
    // ── Function management ──────────────────────────────────────────────────
 
    Function* add_function(std::string name, TypePtr fn_type,
                           bool is_async = false)
    {
        for (auto& fn : functions) {
            if (fn->name == name) return fn.get();
        }
        auto fn = std::make_shared<Function>(std::move(name), std::move(fn_type));
        fn->is_async = is_async;
        fn->parent_module = this;
        functions.push_back(std::move(fn));
        return functions.back().get();
    }
 
    Function* find_function(const std::string& name) const
    {
        for (auto& fn : functions)
            if (fn->name == name) return fn.get();
        return nullptr;
    }
 
    // ── Global value management ──────────────────────────────────────────────
 
    Value* add_global(std::shared_ptr<Value> val)
    {
        globals.push_back(std::move(val));
        return globals.back().get();
    }
 
    Value* find_global(const std::string& name) const
    {
        for (auto& g : globals)
            if (g->name == name) return g.get();
        return nullptr;
    }
};
 
} // namespace ir