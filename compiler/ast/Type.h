#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <cassert>
#include <functional>

enum class TyKind;
using Dim = std::variant<int, std::string>;
inline std::string dim_str(const Dim& d) {
    return std::holds_alternative<int>(d) ? std::to_string(std::get<int>(d)) : std::get<std::string>(d);
}
inline bool dim_compat(const Dim& a, const Dim& b) {
    if (std::holds_alternative<std::string>(a)) return true;
    if (std::holds_alternative<std::string>(b)) return true;
    return std::get<int>(a) == std::get<int>(b);
}
inline bool dims_compat(const std::vector<Dim>& a, const std::vector<Dim>& b) {
    if (a.empty() || b.empty()) return true;   // unknown rank — defer to runtime
    if (a.size() != b.size())   return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!dim_compat(a[i], b[i])) return false;
    return true;
}

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
        // async
        Task,       // args[0]        = inner value type T
                    // Task<T> is the handle returned by `spawn { expr }`.
                    // `await task` unwraps Task<T> back to T.
                    // Only legal inside async fn or spawn body.
        // user-defined / generic
        Named,      // type_name = struct name, e.g. "Point"
        Var,        // type_name = generic param name, e.g. "T"
        // unresolved
        Infer,      // unknown — to be resolved by type inference
    } kind;

    std::vector<TypePtr> args;        // inner types (see table above)
    std::string          type_name;   // Named / Var only
    std::vector<Dim>     shape;       // Tensor only: dimension sizes

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

    bool    is_task()     const { return kind == Kind::Task; }
    TypePtr inner_type()  const { return (!args.empty()) ? args[0] : infer(); }

    bool operator==(const Type& o) const {
        if (kind == Kind::Infer || o.kind == Kind::Infer) return true;
        if (kind != o.kind)                               return false;
        if (kind == Kind::Named || kind == Kind::Var)
            return type_name == o.type_name;
        if (kind == Kind::Tensor && !dims_compat(shape, o.shape))     return false;
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
                        out += dim_str(shape[i]);
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
            case Kind::Task:  return "Task<"  + s(args.empty() ? nullptr : args[0]) + ">";
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
        std::vector<Dim> tensor_shape = {});

    static TypePtr fromTyKind(
        TyKind tk,
        const std::optional<std::string>& type_name, 
        std::vector<TypePtr> inner_args = {},
        std::vector<Dim> tensor_shape = {})
    {
        return fromTyKind(tk, type_name.value_or(""), std::move(inner_args), std::move(tensor_shape));
    }
};

inline bool type_compat(const TypePtr& a, const TypePtr& b) {
    if (!a || !b)      return true;
    if (a->is_infer()) return true;
    if (b->is_infer()) return true;
    return *a == *b;
}

struct BuiltinEntry {
    Type::Kind   receiver_kind;
    std::string  member;
    std::function<TypePtr(const TypePtr&)> make_type;
};

inline const std::vector<BuiltinEntry>& builtin_method_table() {
    static const std::vector<BuiltinEntry> tbl = {
        { Type::Kind::Tensor, "shape",         [](const TypePtr&)   { return Type::array(Type::i32()); } },
        { Type::Kind::Tensor, "rank",          [](const TypePtr&)   { return Type::i32(); } },
        { Type::Kind::Tensor, "size",          [](const TypePtr&)   { return Type::i32(); } },
        { Type::Kind::Tensor, "dtype",         [](const TypePtr&)   { return Type::str_(); } },
        { Type::Kind::Tensor, "requires_grad", [](const TypePtr&)   { return Type::bool_(); } },
        { Type::Kind::Tensor, "T",             [](const TypePtr& r) { return r; } },
        { Type::Kind::Tensor, "grad",          [](const TypePtr& r) { return Type::tensor(r->elem_type()); } },
        { Type::Kind::Tensor, "item",          [](const TypePtr& r) { return r->elem_type(); } },
        { Type::Kind::Tensor, "sum",           [](const TypePtr& r) { return Type::fn({}, r->elem_type()); } },
        { Type::Kind::Tensor, "mean",          [](const TypePtr& r) { return Type::fn({}, r->elem_type()); } },
        { Type::Kind::Tensor, "min",           [](const TypePtr& r) { return Type::fn({}, r->elem_type()); } },
        { Type::Kind::Tensor, "max",           [](const TypePtr& r) { return Type::fn({}, r->elem_type()); } },
        { Type::Kind::Tensor, "prod",          [](const TypePtr& r) { return Type::fn({}, r->elem_type()); } },
        { Type::Kind::Tensor, "flatten",       [](const TypePtr& r) { return Type::fn({}, r); } },
        { Type::Kind::Tensor, "contiguous",    [](const TypePtr& r) { return Type::fn({}, r); } },
        { Type::Kind::Tensor, "clone",         [](const TypePtr& r) { return Type::fn({}, r); } },
        { Type::Kind::Tensor, "detach",        [](const TypePtr& r) { return Type::fn({}, r); } },

        { Type::Kind::Array,  "len",           [](const TypePtr&)   { return Type::i32(); } },
        { Type::Kind::Array,  "is_empty",      [](const TypePtr&)   { return Type::bool_(); } },
        { Type::Kind::Array,  "push",          [](const TypePtr& r) { return Type::fn({r->elem_type()}, Type::void_()); } },
        { Type::Kind::Array,  "pop",           [](const TypePtr& r) { return Type::fn({}, r->elem_type()); } },

        { Type::Kind::Map,    "len",           [](const TypePtr&)   { return Type::i32(); } },
        { Type::Kind::Map,    "is_empty",      [](const TypePtr&)   { return Type::bool_(); } },
        { Type::Kind::Map,    "keys",          [](const TypePtr& r) { return Type::array(r->key_type()); } },
        { Type::Kind::Map,    "values",        [](const TypePtr& r) { return Type::array(r->val_type()); } },
        { Type::Kind::Map,    "contains",      [](const TypePtr& r) { return Type::fn({r->key_type()}, Type::bool_()); } },
        { Type::Kind::Map,    "get",           [](const TypePtr& r) { return Type::fn({r->key_type()}, r->val_type()); } },
        { Type::Kind::Map,    "insert",        [](const TypePtr& r) { return Type::fn({r->key_type(), r->val_type()}, Type::void_()); } },
        { Type::Kind::Map,    "remove",        [](const TypePtr& r) { return Type::fn({r->key_type()}, Type::void_()); } },

        { Type::Kind::Str,    "len",           [](const TypePtr&)   { return Type::i32(); } },
        { Type::Kind::Str,    "is_empty",      [](const TypePtr&)   { return Type::bool_(); } },
        { Type::Kind::Str,    "to_upper",      [](const TypePtr&)   { return Type::fn({}, Type::str_()); } },
        { Type::Kind::Str,    "to_lower",      [](const TypePtr&)   { return Type::fn({}, Type::str_()); } },
        { Type::Kind::Str,    "trim",          [](const TypePtr&)   { return Type::fn({}, Type::str_()); } },
        { Type::Kind::Str,    "contains",      [](const TypePtr&)   { return Type::fn({Type::str_()}, Type::bool_()); } },
        { Type::Kind::Str,    "split",         [](const TypePtr&)   { return Type::array(Type::str_()); } },
        { Type::Kind::Str,    "parse_i32",     [](const TypePtr&)   { return Type::i32(); } },
        { Type::Kind::Str,    "parse_f32",     [](const TypePtr&)   { return Type::f32(); } },

        { Type::Kind::Task,   "is_done",       [](const TypePtr&)   { return Type::bool_(); } },
        { Type::Kind::Task,   "cancel",        [](const TypePtr&)   { return Type::fn({}, Type::void_()); } },
    };
    return tbl;
}

inline TypePtr lookup_builtin(const TypePtr& receiver, const std::string& member) {
    if (!receiver) return nullptr;
    for (auto& e : builtin_method_table())
        if (e.receiver_kind == receiver->kind && e.member == member)
            return e.make_type(receiver);
    return nullptr;
}

inline std::string builtin_members_for(Type::Kind k) {
    std::string out;
    for (auto& e : builtin_method_table())
        if (e.receiver_kind == k) { if (!out.empty()) out += ", "; out += e.member; }
    return out;
}

