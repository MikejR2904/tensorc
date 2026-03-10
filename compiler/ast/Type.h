#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <cassert>
#include "ASTNode.h"

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct Type
{
    enum class Kind
    {
        // primitives
        I32, I64,
        F32, F64,
        Bool,
        Str,
        Void,
        // parameterised collections
        Array,      // args[0]        = element type
        Tensor,     // args[0]        = element type;  shape = dimension sizes
        Map,        // args[0]=key    args[1]=value
        Set,        // args[0]        = element type
        Queue,      // args[0]        = element type
        Stack,      // args[0]        = element type
        Tuple,      // args[0..N]     = element types  (heterogeneous)
        // callables
        Fn,         // args[0..N-1]   = parameter types
                    // args[N]        = return type  (always last)
        // user-defined / generic
        Named,      // type_name = struct name, e.g. "Point"
        Var,        // type_name = generic param name, e.g. "T"
        // unresolved
        Infer,      // unknown — to be resolved by type inference
    } kind;

    std::vector<TypePtr> args;        // inner types (see table above)
    std::string          type_name;   // Named / Var only
    std::vector<int>     shape;       // Tensor only: dimension sizes

    explicit Type(Kind k) : kind(k) {}

    static TypePtr i32()   { return std::make_shared<Type>(Kind::I32);  }
    static TypePtr i64()   { return std::make_shared<Type>(Kind::I64);  }
    static TypePtr f32()   { return std::make_shared<Type>(Kind::F32);  }
    static TypePtr f64()   { return std::make_shared<Type>(Kind::F64);  }
    static TypePtr bool_() { return std::make_shared<Type>(Kind::Bool); }
    static TypePtr str_()  { return std::make_shared<Type>(Kind::Str);  }
    static TypePtr void_() { return std::make_shared<Type>(Kind::Void); }
    static TypePtr infer() { return std::make_shared<Type>(Kind::Infer);}

    static TypePtr array(TypePtr elem) {
        auto t = std::make_shared<Type>(Kind::Array);
        t->args = { std::move(elem) };
        return t;
    }

    static TypePtr tensor(TypePtr elem, std::vector<int> shape = {}) {
        auto t = std::make_shared<Type>(Kind::Tensor);
        t->args  = { std::move(elem) };
        t->shape = std::move(shape);
        return t;
    }

    static TypePtr map(TypePtr key, TypePtr val) {
        auto t = std::make_shared<Type>(Kind::Map);
        t->args = { std::move(key), std::move(val) };
        return t;
    }

    static TypePtr set(TypePtr elem) {
        auto t = std::make_shared<Type>(Kind::Set);
        t->args = { std::move(elem) };
        return t;
    }

    static TypePtr queue(TypePtr elem) {
        auto t = std::make_shared<Type>(Kind::Queue);
        t->args = { std::move(elem) };
        return t;
    }

    static TypePtr stack(TypePtr elem) {
        auto t = std::make_shared<Type>(Kind::Stack);
        t->args = { std::move(elem) };
        return t;
    }

    static TypePtr tuple(std::vector<TypePtr> elems) {
        auto t = std::make_shared<Type>(Kind::Tuple);
        t->args = std::move(elems);
        return t;
    }

    static TypePtr fn(std::vector<TypePtr> params, TypePtr ret) {
        auto t = std::make_shared<Type>(Kind::Fn);
        t->args = std::move(params);
        t->args.push_back(std::move(ret));   // return type is always last
        return t;
    }

    static TypePtr named(std::string name) {
        auto t = std::make_shared<Type>(Kind::Named);
        t->type_name = std::move(name);
        return t;
    }

    static TypePtr var(std::string name) {
        auto t = std::make_shared<Type>(Kind::Var);
        t->type_name = std::move(name);
        return t;
    }

    bool is_infer() const { return kind == Kind::Infer; }
    bool is_void()  const { return kind == Kind::Void;  }

    bool is_numeric() const {
        return kind == Kind::I32 || kind == Kind::I64 ||
               kind == Kind::F32 || kind == Kind::F64;
    }

    bool is_bool() const { return kind == Kind::Bool; }

    bool is_collection() const {
        return kind == Kind::Array  || kind == Kind::Tensor ||
               kind == Kind::Map   || kind == Kind::Set    ||
               kind == Kind::Queue || kind == Kind::Stack  ||
               kind == Kind::Tuple;
    }

    TypePtr elem_type() const {
        return (!args.empty()) ? args[0] : infer();
    }

    TypePtr key_type() const { return (args.size() > 0) ? args[0] : infer(); }
    TypePtr val_type() const { return (args.size() > 1) ? args[1] : infer(); }

    TypePtr ret_type() const { return args.empty() ? void_() : args.back(); }

    std::vector<TypePtr> param_types() const {
        if (args.size() < 2) return {};
        return { args.begin(), args.end() - 1 };
    }

    bool operator==(const Type& o) const {
        if (kind == Kind::Infer || o.kind == Kind::Infer) return true;
        if (kind != o.kind)                               return false;
        if (kind == Kind::Named || kind == Kind::Var)
            return type_name == o.type_name;
        if (kind == Kind::Tensor && shape != o.shape)     return false;
        if (args.size() != o.args.size())                 return false;
        for (size_t i = 0; i < args.size(); ++i)
            if (!(*args[i] == *o.args[i]))                return false;
        return true;
    }

    bool operator!=(const Type& o) const { return !(*this == o); }

    std::string str() const {
        // Safely stringify a nullable TypePtr child.
        auto s = [](const TypePtr& p) -> std::string {
            return p ? p->str() : "?";
        };

        switch (kind) {
            case Kind::I32:   return "i32";
            case Kind::I64:   return "i64";
            case Kind::F32:   return "f32";
            case Kind::F64:   return "f64";
            case Kind::Bool:  return "bool";
            case Kind::Str:   return "str";
            case Kind::Void:  return "void";
            case Kind::Infer: return "<infer>";
            case Kind::Named: return type_name;
            case Kind::Var:   return type_name;

            case Kind::Array:
                return "Array<" + s(args.empty() ? nullptr : args[0]) + ">";

            case Kind::Tensor: {
                std::string out = "Tensor<" + s(args.empty() ? nullptr : args[0]);
                if (!shape.empty()) {
                    out += ", [";
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (i) out += ",";
                        out += std::to_string(shape[i]);
                    }
                    out += "]";
                }
                return out + ">";
            }

            case Kind::Map:
                return "Map<" + s(args.size() > 0 ? args[0] : nullptr)
                     + ", " + s(args.size() > 1 ? args[1] : nullptr) + ">";

            case Kind::Set:   return "Set<"   + s(args.empty() ? nullptr : args[0]) + ">";
            case Kind::Queue: return "Queue<" + s(args.empty() ? nullptr : args[0]) + ">";
            case Kind::Stack: return "Stack<" + s(args.empty() ? nullptr : args[0]) + ">";

            case Kind::Tuple: {
                std::string out = "Tuple<";
                for (size_t i = 0; i < args.size(); ++i) {
                    if (i) out += ", ";
                    out += s(args[i]);
                }
                return out + ">";
            }

            case Kind::Fn: {
                std::string out = "fn(";
                auto params = param_types();
                for (size_t i = 0; i < params.size(); ++i) {
                    if (i) out += ", ";
                    out += s(params[i]);
                }
                return out + ") -> " + s(ret_type());
            }

            default: return "?";
        }
    }

    static TypePtr fromTyKind(
        TyKind tk,
        const std::string& type_name = "",
        std::vector<TypePtr> inner_args = {},
        std::vector<int> tensor_shape = {});

    static TypePtr fromTyKind(
        TyKind                            tk,
        const std::optional<std::string>& type_name,
        std::vector<TypePtr>              inner_args   = {},
        std::vector<int>                  tensor_shape  = {})
    {
        return fromTyKind(tk,
                          type_name.value_or(""),
                          std::move(inner_args),
                          std::move(tensor_shape));
    }
};

inline bool type_compat(const TypePtr& a, const TypePtr& b) {
    if (!a || !b)      return true;
    if (a->is_infer()) return true;
    if (b->is_infer()) return true;
    return *a == *b;
}

inline TypePtr Type::fromTyKind(
    TyKind              tk,
    const std::string&   type_name,
    std::vector<TypePtr> inner_args,
    std::vector<int>    tensor_shape)
{
    auto arg = [&](size_t i) -> TypePtr {
        return (i < inner_args.size() && inner_args[i]) ? inner_args[i] : infer();
    };
    switch (tk) {
        case TyKind::I32:     return i32();
        case TyKind::I64:     return i64();
        case TyKind::F32:     return f32();
        case TyKind::F64:     return f64();
        case TyKind::Bool:    return bool_();
        case TyKind::Str:     return str_();
        case TyKind::Void:    return void_();
        case TyKind::Infer:   return infer();
        case TyKind::FnType:  {
            if (inner_args.empty()) return fn({}, void_());
            TypePtr ret = inner_args.back();
            inner_args.pop_back();
            return fn(std::move(inner_args), std::move(ret));
        }
        case TyKind::Generic: return var(type_name.empty() ? "T" : type_name);
        case TyKind::UserDef: return named(type_name);
        case TyKind::Array:   return array(arg(0));
        case TyKind::Tensor:  {
            TypePtr elem = (inner_args.empty() || !inner_args[0])
                         ? f32() : inner_args[0];
            return tensor(std::move(elem), std::move(tensor_shape));
        }
        case TyKind::Map:     return map(arg(0), arg(1));
        case TyKind::Set:     return set(arg(0));
        case TyKind::Queue:   return queue(arg(0));
        case TyKind::Stack:   return stack(arg(0));
        case TyKind::Tuple:   return tuple(std::move(inner_args));
        default:              return infer();
    }
}
