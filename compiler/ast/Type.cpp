#include "Type.h"
#include "ASTNode.h"

TypePtr Type::fromTyKind(
    TyKind tk,
    const std::string& type_name,
    std::vector<TypePtr> inner_args,
    std::vector<Dim> tensor_shape) 
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