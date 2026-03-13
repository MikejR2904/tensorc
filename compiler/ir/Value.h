#pragma once
 
#include "../ast/Type.h"
#include <string>
#include <vector>
#include <memory>
 
namespace ir {
 
// ─── Forward declarations ────────────────────────────────────────────────────
 
struct Value;
struct Instruction;
struct BasicBlock;
struct Function;
struct IRModule;
 
using ValuePtr = std::shared_ptr<Value>;
using InstPtr  = std::shared_ptr<Instruction>;
using BBPtr    = std::shared_ptr<BasicBlock>;
using FuncPtr  = std::shared_ptr<Function>;
 
// ─── Value ───────────────────────────────────────────────────────────────────
 
/// The base class for all IR values.
/// Subclassed by: Instruction, Argument, Constant, GlobalValue.
struct Value
{
    std::string name;    ///< SSA name, e.g. "%3", "%result", "@my_fn"
    TypePtr     type;    ///< Resolved type from sema
 
    /// All instructions that consume this value (use-def chain).
    std::vector<Instruction*> uses;
 
    Value(std::string name, TypePtr type)
        : name(std::move(name)), type(std::move(type)) {}
 
    virtual ~Value() = default;
 
    void add_use(Instruction* user)    { uses.push_back(user); }
    bool has_uses()              const { return !uses.empty(); }
 
    virtual bool is_constant()   const { return false; }
    virtual bool is_argument()   const { return false; }
    virtual bool is_instruction()const { return false; }
};
 
// ─── Argument ────────────────────────────────────────────────────────────────
 
/// A function parameter — appears as a Value at the entry block.
struct Argument : Value
{
    size_t index;   ///< Position in the parameter list
 
    Argument(std::string name, TypePtr type, size_t index)
        : Value(std::move(name), std::move(type)), index(index) {}
 
    bool is_argument() const override { return true; }
};
 
// ─── Constants ───────────────────────────────────────────────────────────────
 
struct Constant : Value
{
    Constant(std::string name, TypePtr type)
        : Value(std::move(name), std::move(type)) {}
 
    bool is_constant() const override { return true; }
};
 
struct ConstantInt : Constant
{
    int64_t value;
 
    ConstantInt(int64_t v, TypePtr type)
        : Constant("i" + std::to_string(v), std::move(type)), value(v) {}
};
 
struct ConstantFloat : Constant
{
    double value;
 
    ConstantFloat(double v, TypePtr type)
        : Constant("f" + std::to_string(v), std::move(type)), value(v) {}
};
 
struct ConstantBool : Constant
{
    bool value;
 
    ConstantBool(bool v)
        : Constant(v ? "true" : "false", Type::bool_()), value(v) {}
};
 
struct ConstantString : Constant
{
    std::string value;
 
    ConstantString(std::string v)
        : Constant("str", Type::str_()), value(std::move(v)) {}
};
 
/// A tensor constant (e.g. from a literal initialiser).
struct ConstantTensor : Constant
{
    std::vector<float>   data;
    std::vector<int64_t> shape;
 
    ConstantTensor(std::vector<float> data,
                   std::vector<int64_t> shape,
                   TypePtr type)
        : Constant("tensor_const", std::move(type))
        , data(std::move(data))
        , shape(std::move(shape)) {}
};
 
} // namespace ir