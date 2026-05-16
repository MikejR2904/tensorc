#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <cassert>
#include <functional>

enum class TyKind; // forward declaration to avoid circular dependency with ASTNode.h

// Type system for type checking and inference. Types are represented as a tree structure with a kind (e.g. Tensor, Map) and inner types (e.g. element type, key/value types).
// Dim is a helper type for tensor dimensions, which can be either a concrete integer (e.g. 3) or a symbolic name (e.g. 'N') that will be resolved at runtime.
using Dim = std::variant<int, std::string>;
inline std::string dim_str(const Dim& d) { return std::holds_alternative<int>(d) ? std::to_string(std::get<int>(d)) : std::get<std::string>(d); }

inline bool dim_compat(const Dim& a, const Dim& b) {
    // Two dimensions are compatible if they are equal integers, or if either is a symbolic name (which could match any size at runtime).
    if (std::holds_alternative<std::string>(a)) return true; // if one is symbolic, it's compatible with the other
    if (std::holds_alternative<std::string>(b)) return true;
    return std::get<int>(a) == std::get<int>(b);
}

inline bool dims_compat(const std::vector<Dim>& a, const std::vector<Dim>& b) {
    if (a.empty() || b.empty()) return true; // unknown rank; defer to runtime
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) // check each dimension pairwise
        if (!dim_compat(a[i], b[i])) return false;
    // Note that here we don't check whether the string names match, because they are just symbolic placeholders that could be any size at runtime. 
    // The important thing is that they are consistent within the same type (e.g. Tensor[N, N] is compatible with Tensor[3, 3], but not with Tensor[3, 4]). Tensor[N, M] can be compatible with Tensor[X, Y], which we will resolve later during runtime whether they are actually equal.
    return true;
}

struct Type; // forward declaration
using TypePtr = std::shared_ptr<Type>; // shared_ptr for easier handling of recursive types (e.g. Fn that contains itself)

struct Type
{
    enum class Kind // Resembles/recreates TyKind; the difference is that TyKind is a simple enum used in the parser to represent type annotations, while Type::Kind is a richer structure used in the semantic analyzer and IR generation to represent fully resolved types with their inner structure (e.g. element types, shape).
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
        // async
        Task,       // args[0]        = inner value type T
        // user-defined / generic
        Named,      // type_name = struct name, e.g. "Point"
        Var,        // type_name = generic param name, e.g. "T"
        // unresolved
        Infer,      // unknown; to be resolved by type inference
    } kind;

    std::vector<TypePtr> args; // inner types (see table above)
    std::string type_name; // Named / Var only
    std::vector<Dim> shape; // Tensor only: dimension sizes

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

    static TypePtr tensor(TypePtr elem, std::vector<Dim> shape = {}) {
        auto t = std::make_shared<Type>(Kind::Tensor);
        t->args  = { std::move(elem) };
        t->shape = std::move(shape);
        return t;
    }

    static TypePtr tensor(TypePtr elem, std::vector<int> int_shape) {
        std::vector<Dim> shape;
        shape.reserve(int_shape.size());
        for (int d : int_shape) shape.emplace_back(d);
        return tensor(std::move(elem), std::move(shape));
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

    static TypePtr task(TypePtr inner) {
        auto t = std::make_shared<Type>(Kind::Task);
        t->args = { std::move(inner) };
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

    bool is_numeric() const { return kind == Kind::I32 || kind == Kind::I64 || kind == Kind::F32 || kind == Kind::F64; }
    bool is_float() const { return kind == Kind::F32 || kind == Kind::F64; }
    bool is_bool() const { return kind == Kind::Bool; }

    bool is_collection() const {
        return kind == Kind::Array  || kind == Kind::Tensor ||
               kind == Kind::Map   || kind == Kind::Set    ||
               kind == Kind::Queue || kind == Kind::Stack  ||
               kind == Kind::Tuple;
    }

    TypePtr elem_type() const { return (!args.empty()) ? args[0] : infer(); }
    TypePtr key_type() const { return (args.size() > 0) ? args[0] : infer(); }
    TypePtr val_type() const { return (args.size() > 1) ? args[1] : infer(); }
    TypePtr ret_type() const { return args.empty() ? void_() : args.back(); }

    std::vector<TypePtr> param_types() const {
        if (args.size() < 2) return {};
        return { args.begin(), args.end() - 1 };
    }

    bool is_task() const { return kind == Kind::Task; }
    TypePtr inner_type() const { return (!args.empty()) ? args[0] : infer(); }

    bool operator==(const Type& o) const {
        if (kind == Kind::Infer || o.kind == Kind::Infer) return true;
        if (kind != o.kind) return false;
        if (kind == Kind::Named || kind == Kind::Var) return type_name == o.type_name;
        if (kind == Kind::Tensor && !dims_compat(shape, o.shape)) return false;
        if (args.size() != o.args.size()) return false;
        for (size_t i = 0; i < args.size(); ++i)
            if (!(*args[i] == *o.args[i])) return false;
        return true;
    }

    bool operator!=(const Type& o) const { return !(*this == o); }

    std::string str() const {
        // Stringify a nullable TypePtr child.
        auto s = [](const TypePtr& p) -> std::string { return p ? p->str() : "?"; };

        switch (kind) {
            case Kind::I32: return "i32";
            case Kind::I64: return "i64";
            case Kind::F32: return "f32";
            case Kind::F64: return "f64";
            case Kind::Bool: return "bool";
            case Kind::Str: return "str";
            case Kind::Void: return "void";
            case Kind::Infer: return "<infer>";
            case Kind::Named: return type_name;
            case Kind::Var: return type_name;
            case Kind::Array:
                return "Array<" + s(args.empty() ? nullptr : args[0]) + ">";
            case Kind::Tensor: {
                std::string out = "Tensor<" + s(args.empty() ? nullptr : args[0]);
                if (!shape.empty()) {
                    out += ", [";
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (i) out += ",";
                        out += dim_str(shape[i]);
                    }
                    out += "]";
                }
                return out + ">";
            }
            case Kind::Map: return "Map<" + s(args.size() > 0 ? args[0] : nullptr) + ", " + s(args.size() > 1 ? args[1] : nullptr) + ">";
            case Kind::Set: return "Set<"   + s(args.empty() ? nullptr : args[0]) + ">";
            case Kind::Queue: return "Queue<" + s(args.empty() ? nullptr : args[0]) + ">";
            case Kind::Stack: return "Stack<" + s(args.empty() ? nullptr : args[0]) + ">";
            case Kind::Task: return "Task<"  + s(args.empty() ? nullptr : args[0]) + ">";
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

    // To bridge between TyKind from Lexer to Type, we use this method to convert between parallel type representations. The TyKind enum is a simpler representation used during parsing to represent type annotations, while the Type struct is a richer representation used during semantic analysis and IR generation to represent fully resolved types with their inner structure (e.g. element types, shape). The fromTyKind method takes a TyKind and optional additional information (e.g. user type name for Named/Var, inner types for collections) and constructs the corresponding TypePtr.
    static TypePtr fromTyKind(TyKind tk, const std::string& type_name = "", std::vector<TypePtr> inner_args = {}, std::vector<Dim> tensor_shape = {});
    static TypePtr fromTyKind(TyKind tk, const std::optional<std::string>& type_name,  std::vector<TypePtr> inner_args = {}, std::vector<Dim> tensor_shape = {})
    {
        return fromTyKind(tk, type_name.value_or(""), std::move(inner_args), std::move(tensor_shape));
    }
};

// Check if two types are compatible for assignment or comparison. This is a relaxed check that allows for type inference to fill in the gaps. For example, if either type is 'Infer', we consider them compatible, because 'Infer' can unify with any type. Otherwise, they must be exactly equal.
// Used in SemanticAnalyzer.h 
inline bool type_compat(const TypePtr& a, const TypePtr& b) {
    if (!a || !b) return true;
    if (a->is_infer()) return true;
    if (b->is_infer()) return true;
    return *a == *b;
}

