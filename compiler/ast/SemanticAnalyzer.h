#pragma once

#include "ASTNode.h"
#include "Type.h"
#include "SymbolTable.h"
#include "../io/ImportRegistry.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>

struct StructDef {
    std::string name;
    std::unordered_map<std::string, TypePtr> fields;
    
};

class SemanticAnalyzer {
public:
    explicit SemanticAnalyzer(ImportRegistry imports = ImportRegistry::with_builtins()) 
        : symb_tab(), import_reg(std::move(imports)) {}

    SemanticAnalyzer() : symb_tab() {}
    void validate(Program& program) {
        try {
            for (auto& stmt : program.stmts)
                register_top_level(stmt.get());
            validate_structs();
            for (auto& stmt : program.stmts) {
                validate_stmt(stmt.get());
            }
        } catch (const std::runtime_error& e) {
            std::cerr << "Semantic Error: " << e.what() << std::endl;
            throw; 
        }
    }

private:
    SymbolTable symb_tab;
    ImportRegistry import_reg;
    std::unordered_map<std::string, std::string> active_imports;
    std::unordered_map<std::string, StructDef> struct_registry;
    TypePtr expected_ret_ty = Type::void_();
    int iteration_depth = 0;
    int async_depth = 0;
    void register_struct(const StructDef& def)
    {
        struct_registry[def.name] = std::move(def);
    }

    void register_top_level(Stmt* stmt)
    {
        if (stmt->kind.tag == StmtKind::Tag::Struct)
        {
            StructDef def;
            def.name = stmt->kind.struct_name;
            for (auto& f : stmt->kind.struct_fields) {
                TypePtr field_ty = Type::fromTyKind(f.ty, f.user_type_name);
                def.fields[f.name] = std::move(field_ty);
            }
            register_struct(def);
        }
        if (stmt->kind.tag == StmtKind::Tag::Func && stmt->kind.func.has_value())
        {
            auto& func = stmt->kind.func.value();
            std::vector<TypePtr> p_tys;
            for (auto& p : func.params)
                p_tys.push_back(Type::fromTyKind(p.ty_kind(), p.user_type_name()));
            TypePtr ret = Type::fromTyKind(func.ident.ty_kind(), func.ident.user_type_name());
            symb_tab.define(
                Symbol(func.ident.name(),
                    Type::fn(std::move(p_tys), std::move(ret)),
                    IdentCtx::FuncDef,
                    func.ident.pos()));
        }
    }

    // --- Statement Validation ---
    TypePtr validate_stmt(Stmt* stmt) {
        if (!stmt) return Type::void_();
        switch (stmt->kind.tag) {
            case StmtKind::Tag::Let: return validate_let(stmt);
            case StmtKind::Tag::Func: {
                if (stmt->kind.func.has_value()) {
                    return validate_func(stmt->kind.func.value());
                }
                return Type::void_();
            }
            case StmtKind::Tag::Return: {
                TypePtr actual = stmt->kind.ret_expr
                    ? validate_expr(stmt->kind.ret_expr.get())
                    : Type::void_();
                expect_compat(expected_ret_ty, actual, stmt->pos, "Return type mismatch");           
                return actual;
            }
            case StmtKind::Tag::If: return validate_if(stmt);
            case StmtKind::Tag::Else: return validate_compound(stmt->kind.else_body, false);
            case StmtKind::Tag::While: return validate_while(stmt);
            case StmtKind::Tag::For: return validate_for(stmt);
            case StmtKind::Tag::Compound: return validate_compound(stmt->kind.compound, false);
            case StmtKind::Tag::Expr: return validate_expr(stmt->kind.expr.get());
            case StmtKind::Tag::Break:
            case StmtKind::Tag::Continue: {
                if (iteration_depth == 0) {
                    error("Statement must be inside a loop context", stmt->pos);
                }
                return Type::void_();
            }
            case StmtKind::Tag::Match: return validate_match(stmt);
            case StmtKind::Tag::Spawn: return validate_spawn(stmt);
            case StmtKind::Tag::Import: {
                const std::string& mod_name = stmt->kind.import_path;
                const std::string& alias = stmt->kind.import_alias.empty() ? mod_name : stmt->kind.import_alias;
                if (!import_reg.has_module(mod_name)) {
                    error("Module '" + mod_name + "' not found in registry", stmt->pos);
                }
                active_imports[alias] = mod_name;
                return Type::void_();
            }
            default: return Type::void_();
        }
        return Type::void_(); 
    }

    TypePtr validate_let(Stmt* stmt) { // let x [: T] = expr
        // Validate RHS first to prevent recursive definitions like 'let x = x'
        TypePtr rhs_ty = validate_expr(stmt->kind.let_expr.get());
        TyKind  ann_k  = stmt->kind.let_ident.ty_kind();
        TypePtr var_ty;
        if (ann_k == TyKind::Infer) {
            var_ty = rhs_ty;
            stmt->kind.let_ident.set_ty_kind(rhs_ty ? tkFromType(rhs_ty) : TyKind::Infer);
        } else {
            bool propagate_elem = (ann_k == TyKind::Array  ||
                                   ann_k == TyKind::Tensor ||
                                   ann_k == TyKind::Set    ||
                                   ann_k == TyKind::Queue  ||
                                   ann_k == TyKind::Stack);
            std::vector<TypePtr> inner;
            if (propagate_elem && rhs_ty) inner.push_back(rhs_ty);
            var_ty = Type::fromTyKind(
                ann_k,
                stmt->kind.let_ident.user_type_name(),  // struct name if UserDef
                std::move(inner));
            expect_compat(var_ty, rhs_ty, stmt->pos, "Variable '" + stmt->kind.let_ident.name() + "' type mismatch");
        }
        Symbol sym(stmt->kind.let_ident.name(), var_ty, IdentCtx::Def, stmt->pos);
        if (var_ty && var_ty->kind == Type::Kind::Tensor) {
            const auto& utn = stmt->kind.let_ident.user_type_name();
            sym.requires_grad = (!utn->empty() && utn == "@grad");
        }
        symb_tab.define(sym);
        return var_ty;
    }

    TypePtr validate_func(Func& func) {
        Symbol* existing = symb_tab.lookup(func.ident.name());
        std::vector<TypePtr> p_tys;
        for (auto& p : func.params) {
            p_tys.push_back(Type::fromTyKind(p.ty_kind(), p.user_type_name()));
        }
        TypePtr ret = Type::fromTyKind(func.ident.ty_kind(), func.ident.user_type_name());
        TypePtr fn_ty = Type::fn(p_tys, ret);
        if (!existing) {
            symb_tab.define(Symbol(func.ident.name(), fn_ty, IdentCtx::FuncDef, func.ident.pos()));
        }
        symb_tab.pushScope();
        TypePtr old_expect = expected_ret_ty;
        expected_ret_ty = ret;
        if (func.is_async) async_depth++;
        // Register parameters in function scope
        for (auto& param : func.params) {
            symb_tab.define(Symbol(param.name(), Type::fromTyKind(param.ty_kind(), param.user_type_name()), IdentCtx::Param, param.pos()));
        }
        TypePtr body_ret_ty = validate_compound(func.body, true);
        if (func.is_async) async_depth--;
        // Ensure return type matches
        if (!ret->is_void())
            expect_compat(ret, body_ret_ty, func.ident.pos(), "Function '" + func.ident.name() + "' return type mismatch");
        expected_ret_ty = old_expect;
        symb_tab.popScope();
        return fn_ty;
    }

    // --- Expression Validation ---

    TypePtr validate_expr(Expr* expr)
    {
        if (!expr) return Type::void_();
        TypePtr ty = validate_expr_inner(expr);
        expr->resolved_type = ty;
        return ty;
    }

    TypePtr validate_expr_inner(Expr* expr) {
        switch (expr->kind.tag) {
            case ExprKind::Tag::Lit: return validate_lit(expr->kind.lit);
            case ExprKind::Tag::Id: {
                const std::string& name = expr->kind.id.name();
                if (name == "_") return Type::infer();
                Symbol* s = symb_tab.lookup(name);
                if (!s) error("Undefined identifier: " + name, expr->pos);
                // Update AST node with the resolved type for the code generator
                return s->type;
            }
            case ExprKind::Tag::Unary: return validate_unary(expr->kind.unary_op, expr->kind.operand.get(), expr->pos);
            case ExprKind::Tag::Binary: return validate_bin_op(expr->kind.bin_op, expr->kind.lhs.get(), expr->kind.rhs.get(), expr->pos);
            case ExprKind::Tag::Assign: {
                if (expr->kind.lhs && expr->kind.lhs->kind.tag == ExprKind::Tag::Id) {
                    const std::string& lname = expr->kind.lhs->kind.id.name();
                    Symbol* ls = symb_tab.lookup(lname);
                    if (ls && !ls->is_mutable)
                        error("Cannot assign to immutable variable '" + lname + "'", expr->pos);
                }
                TypePtr l = validate_expr(expr->kind.lhs.get());
                TypePtr r = validate_expr(expr->kind.rhs.get());
                expect_compat(l, r, expr->pos, "Assignment type mismatch");
                return Type::void_();
            }
            case ExprKind::Tag::Call: return validate_call(expr);
            case ExprKind::Tag::Pipe: return validate_pipe(expr);
            case ExprKind::Tag::Grad: { // grad(loss, params)
                TypePtr loss  = validate_expr(expr->kind.grad_loss.get());
                TypePtr param = validate_expr(expr->kind.grad_params.get());
                if (!loss->is_infer() && !loss->is_numeric()) {
                    error("grad() loss must be f32 or f64 (got " + loss->str() + ")", expr->pos);
                }
                if (!param->is_infer() && param->kind != Type::Kind::Tensor && param->kind != Type::Kind::Var) {
                    error("grad() params must be Tensor (got " + param->str() + ")", expr->pos);
                }
                if (expr->kind.grad_params->kind.tag == ExprKind::Tag::Id) {
                    const std::string& pname =
                        expr->kind.grad_params->kind.id.name();
                    Symbol* psym = symb_tab.lookup(pname);
                    if (psym && !psym->requires_grad)
                        error("grad() called on '" + pname +
                              "' which does not require gradients. "
                              "Declare it with @grad to enable gradient tracking.",
                              expr->pos);
                }
                return param; 
            }
            case ExprKind::Tag::If: {
                TypePtr cond_ty = validate_expr(expr->kind.condition.get());
                expect_compat(Type::bool_(), cond_ty, expr->pos, "if condition must be bool");
                TypePtr then_ty = validate_expr(expr->kind.then_branch.get());
                if (expr->kind.else_branch)
                {
                    TypePtr else_ty = validate_expr(expr->kind.else_branch.get());
                    expect_compat(then_ty, else_ty, expr->pos, "if/else branches must have the same type");
                }
                return then_ty;
            }
            case ExprKind::Tag::Block: return validate_compound(expr->kind.block, false);
            case ExprKind::Tag::TensorLit: {
                TypePtr elem = Type::f32();
                std::vector<Dim> shape;
                if (expr->kind.generic_params.has_value()) {
                    auto& gp = *expr->kind.generic_params;
                    if (!gp.type_params.empty())
                        elem = Type::fromTyKind(gp.type_params[0]);
                    shape.reserve(gp.shape.size());
                    for (int d : gp.shape) shape.emplace_back(d);
                }
                return Type::tensor(elem, shape);
            }
            case ExprKind::Tag::ArrayLit: {
                if (expr->kind.elements.empty()) return Type::array(Type::infer());
                TypePtr first_ty = validate_expr(expr->kind.elements[0].get());
                for (size_t i = 1; i < expr->kind.elements.size(); ++i) {
                    TypePtr current_ty = validate_expr(expr->kind.elements[i].get());
                    expect_compat(first_ty, current_ty, expr->kind.elements[i]->pos, "Array elements must have consistent types");
                }
                return Type::array(first_ty);
            }
            case ExprKind::Tag::SetLit:
            case ExprKind::Tag::QueueLit:
            case ExprKind::Tag::StackLit: {
                TypePtr elem = Type::infer();
                if (!expr->kind.elements.empty())
                    elem = validate_expr(expr->kind.elements[0].get());
                for (size_t i = 1; i < expr->kind.elements.size(); ++i)
                    validate_expr(expr->kind.elements[i].get());
                if (expr->kind.tag == ExprKind::Tag::SetLit) return Type::set(elem);
                if (expr->kind.tag == ExprKind::Tag::QueueLit) return Type::queue(elem);
                if (expr->kind.tag == ExprKind::Tag::StackLit) return Type::stack(elem);
            }
            case ExprKind::Tag::TupleLit: {
                std::vector<TypePtr> elems;
                for (auto& el : expr->kind.elements)
                    elems.push_back(validate_expr(el.get()));
                return Type::tuple(std::move(elems));
            }
            case ExprKind::Tag::MapLit: {
                if (expr->kind.map_pairs.empty()) return Type::map(Type::infer(), Type::infer());
                // Map{"key": value} -> map_pairs is vector<pair<ExprPtr, ExprPtr>>
                TypePtr k_ty = validate_expr(expr->kind.map_pairs[0].first.get());
                TypePtr v_ty = validate_expr(expr->kind.map_pairs[0].second.get());
                for (size_t i = 1; i < expr->kind.map_pairs.size(); ++i) {
                    auto& pair = expr->kind.map_pairs[i];
                    expect_compat(k_ty, validate_expr(pair.first.get()), expr->pos, "Inconsistent Map key");
                    expect_compat(v_ty, validate_expr(pair.second.get()), expr->pos, "Inconsistent Map value");
                }
                return Type::map(k_ty, v_ty);
            }
            case ExprKind::Tag::Index: { // expr[i]
                TypePtr obj_ty = validate_expr(expr->kind.target.get());
                TypePtr idx_ty = validate_expr(expr->kind.index.get());
                if (!idx_ty->is_infer() &&
                    idx_ty->kind != Type::Kind::I32 &&
                    idx_ty->kind != Type::Kind::I64 &&
                    idx_ty->kind != Type::Kind::Var)
                    error("Index subscript must be integer (got " + idx_ty->str() + ")", expr->pos);
                if (!obj_ty || obj_ty->is_infer()) return Type::infer();
                if (obj_ty->kind == Type::Kind::Tensor) {
                    // Rank reduction: rank-1 → scalar, rank-N → Tensor row
                    return (obj_ty->shape.size() == 1)
                        ? obj_ty->elem_type()
                        : Type::tensor(obj_ty->elem_type()); // row, shape unknown
                }
                if (obj_ty->is_collection())
                    return obj_ty->elem_type();
                return Type::infer();
            }
            case ExprKind::Tag::Field: { // expr.member
                TypePtr obj_ty = validate_expr(expr->kind.target.get());
                if (!obj_ty || obj_ty->is_infer()) return Type::infer();
                const std::string& mem = expr->kind.member;
                if (obj_ty->kind == Type::Kind::Tensor) {
                    // Data attributes
                    if (mem == "shape") return Type::array(Type::i32());
                    if (mem == "rank")  return Type::i32();
                    if (mem == "size")  return Type::i32();
                    if (mem == "dtype") return Type::str_();
                    // Reduction methods  →  fn() -> elem_type
                    if (mem == "sum"  || mem == "mean" ||
                        mem == "min"  || mem == "max"  || mem == "prod")
                        return Type::fn({}, obj_ty->elem_type());
                    // Shape manipulation  →  fn() -> Tensor<elem>
                    if (mem == "flatten" || mem == "contiguous" || mem == "clone")
                        return Type::fn({}, obj_ty);
                    if (mem == "T")       // transpose
                        return obj_ty;
                    // Gradient-related
                    if (mem == "grad")
                        return Type::tensor(obj_ty->elem_type());
                    if (mem == "detach")
                        return Type::fn({}, obj_ty);
                    if (mem == "requires_grad")
                        return Type::bool_();
                    // Indexing
                    if (mem == "item")
                        return obj_ty->elem_type();
                    error("Tensor has no attribute '" + mem + "'", expr->pos);
                }
                if (obj_ty->kind == Type::Kind::Array) {
                    if (mem == "len")  return Type::i32();
                    if (mem == "push") return Type::fn({obj_ty->elem_type()}, Type::void_());
                    if (mem == "pop")  return Type::fn({}, obj_ty->elem_type());
                    if (mem == "is_empty") return Type::bool_();
                    error("Array has no attribute '" + mem + "'", expr->pos);
                }
                if (obj_ty->kind == Type::Kind::Map) {
                    if (mem == "len")      return Type::i32();
                    if (mem == "is_empty") return Type::bool_();
                    if (mem == "keys")     return Type::array(obj_ty->key_type());
                    if (mem == "values")   return Type::array(obj_ty->val_type());
                    if (mem == "contains") return Type::fn({obj_ty->key_type()}, Type::bool_());
                    if (mem == "get")      return Type::fn({obj_ty->key_type()}, obj_ty->val_type());
                    if (mem == "insert")   return Type::fn({obj_ty->key_type(), obj_ty->val_type()}, Type::void_());
                    if (mem == "remove")   return Type::fn({obj_ty->key_type()}, Type::void_());
                    error("Map has no attribute '" + mem + "'", expr->pos);
                }
                if (obj_ty->kind == Type::Kind::Str) {
                    if (mem == "len")      return Type::i32();
                    if (mem == "is_empty") return Type::bool_();
                    if (mem == "to_upper" || mem == "to_lower" || mem == "trim")
                        return Type::fn({}, Type::str_());
                    if (mem == "contains") return Type::fn({Type::str_()}, Type::bool_());
                    if (mem == "split")    return Type::array(Type::str_());
                    if (mem == "parse_i32") return Type::i32();
                    if (mem == "parse_f32") return Type::f32();
                    error("Str has no attribute '" + mem + "'", expr->pos);
                }
                if (obj_ty->kind == Type::Kind::Task) {
                    if (mem == "is_done")  return Type::bool_();
                    if (mem == "cancel")   return Type::fn({}, Type::void_());
                    error("Task has no attribute '" + mem + "'", expr->pos);
                }
                if (obj_ty->kind == Type::Kind::Named) {
                    const std::string& sname = obj_ty->type_name;
                    auto it = struct_registry.find(sname);
                    if (it == struct_registry.end())
                        error("Unknown struct type '" + sname + "'", expr->pos);
                    auto fit = it->second.fields.find(mem);
                    if (fit == it->second.fields.end())
                        error("Struct '" + sname + "' has no field '" + mem + "'",
                              expr->pos);
                    return fit->second;
                }

                return Type::infer();
            }
            case ExprKind::Tag::Scope: { // expr::member
                if (!expr->kind.target || expr->kind.target->kind.tag != ExprKind::Tag::Id)
                {
                    validate_expr(expr->kind.target.get());
                    return Type::infer();  // chained scope — future work
                }
                const std::string& alias = expr->kind.target->kind.id.name();
                const std::string& sym   = expr->kind.member;
                std::string canon;
                auto ait = active_imports.find(alias);
                if (ait != active_imports.end()) {
                    canon = ait->second;               // explicit import alias
                } else if (import_reg.has_module(alias)) {
                    canon = alias;                     // built-in, no import needed
                } else {
                    error("Unknown module alias '" + alias + "'. "
                          "Add  import \"" + alias + "\"  (or  import \"<path>\" as " +
                          alias + ")  to bring it into scope.",
                          expr->pos);
                }
                const Symbol* found = import_reg.lookup(canon, sym);
                if (!found) {
                    std::string hint;
                    auto names = import_reg.exported_names(canon);
                    if (names.has_value() && !names->empty()) {
                        hint = " Available: ";
                        for (size_t i = 0; i < names->size(); ++i) {
                            if (i) hint += ", ";
                            hint += (*names)[i];
                        }
                    }
                    error("Module '" + canon + "' (alias '" + alias + "') "
                          "has no exported symbol '" + sym + "'." + hint,
                          expr->pos);
                }
                return found->type;
            }
            case ExprKind::Tag::Range: { // lo..hi
                TypePtr lo = validate_expr(expr->kind.lhs.get());
                TypePtr hi = validate_expr(expr->kind.rhs.get());
                expect_compat(lo, hi, expr->pos, "Range bounds must match");
                return lo;
            }
            case ExprKind::Tag::ChannelSend: { // ch <- val
                validate_expr(expr->kind.channel.get());
                validate_expr(expr->kind.send_val.get());
                return Type::void_();
            }
            case ExprKind::Tag::Await: {
                if (async_depth == 0)
                    error("'await' used outside of an async context. "
                          "Mark the enclosing function 'async fn' or wrap in 'spawn'.",
                          expr->pos);
                TypePtr operand_ty = validate_expr(expr->kind.awaited.get());
                if (!operand_ty || operand_ty->is_infer()) return Type::infer();
                if (operand_ty->kind != Type::Kind::Task)
                    error("'await' requires a Task<T> operand, got " +
                          operand_ty->str() + ". "
                          "Did you forget to 'spawn' the expression?",
                          expr->pos);
                return operand_ty->inner_type();
            }
            case ExprKind::Tag::FnExpr: { // fn(x:T)->T { }
                symb_tab.pushScope();
                TypePtr saved_ret = expected_ret_ty;
                TypePtr ret_ann   = Type::fromTyKind(expr->kind.fn_ret_type);
                expected_ret_ty   = ret_ann;
                std::vector<TypePtr> p_tys;
                for (auto& param : expr->kind.fn_params) {
                    TypePtr pt = Type::fromTyKind(param.second);
                    p_tys.push_back(pt);
                    symb_tab.define(Symbol(param.first, pt, IdentCtx::Param, expr->pos));
                }
                TypePtr body_ty = validate_compound(expr->kind.fn_body, true);
                if (!ret_ann->is_void())
                    expect_compat(ret_ann, body_ty, expr->pos, "Lambda return type mismatch");
                expected_ret_ty = saved_ret;
                symb_tab.popScope();
                return Type::fn(std::move(p_tys), ret_ann);
            }
            case ExprKind::Tag::Match: { // similar to validate_match function
                TypePtr subject_ty = validate_expr(expr->kind.match_subject.get());
                TypePtr match_ret_ty = Type::infer();
                bool first_arm = true;
                for (auto& arm : expr->kind.arms) {
                    symb_tab.pushScope();
                    bool is_wildcard = arm.pattern && arm.pattern->kind.tag == ExprKind::Tag::Id && arm.pattern->kind.id.name() == "_";
                    if (!is_wildcard) {
                        TypePtr pattern_ty = validate_expr(arm.pattern.get());
                        expect_compat(subject_ty, pattern_ty, arm.pos, "Match pattern type mismatch");
                    }
                    if (arm.guard) {
                        TypePtr guard_ty = validate_expr(arm.guard.value().get());
                        expect_compat(Type::bool_(), guard_ty, arm.pos, "Match guard must be bool");
                    }
                    TypePtr arm_ty = arm.hasStmtBody() ? validate_stmt(arm.body_stmt.get()) : validate_expr(arm.body.get());
                    if (first_arm) {
                        match_ret_ty = arm_ty;
                        first_arm = false;
                    } else {
                        expect_compat(match_ret_ty, arm_ty, arm.pos, "Inconsistent match arm types");
                    }
                    symb_tab.popScope();
                }
                return match_ret_ty;
            }
            case ExprKind::Tag::StructLit: {
                const std::string& sname = expr->kind.struct_init_name;
                auto it = struct_registry.find(sname);
                if (it == struct_registry.end()) error("Struct literal uses unknown type '" + sname + "'", expr->pos);
                auto& def = it->second;
                // Check for duplicate field names in the initialiser
                std::unordered_set<std::string> seen_fields;
                for (auto& [fname, fexpr] : expr->kind.struct_init_fields) {
                    if (!seen_fields.insert(fname).second)
                        error("Duplicate field '" + fname +
                              "' in struct literal for '" + sname + "'",
                              expr->pos);
                    auto fit = def.fields.find(fname);
                    if (fit == def.fields.end())
                        error("Field '" + fname +
                              "' does not exist in struct '" + sname + "'",
                              expr->pos);
                    TypePtr actual = validate_expr(fexpr.get());
                    expect_compat(fit->second, actual, fexpr->pos,
                                  "Field '" + fname + "' of '" + sname +
                                  "' type mismatch");
                }
                // Check for missing required fields
                for (auto& [fname, _] : def.fields) {
                    if (seen_fields.find(fname) == seen_fields.end())
                        error("Missing field '" + fname +
                              "' in struct literal for '" + sname + "'",
                              expr->pos);
                }
                return Type::named(sname);
            }
            default:
                return Type::void_();
        }
    }

    TypePtr validate_unary(UnaryOp op, Expr* operand, Position pos)
    {
        TypePtr t = validate_expr(operand);
        if (op == UnaryOp::Not)
        {
            expect_compat(Type::bool_(), t, pos, "! operator requires bool operand");
            return Type::bool_();
        }
        // UnaryOp::Neg — valid for numeric types only.
        if (!t->is_infer() && !t->is_numeric() && t->kind != Type::Kind::Var)
            error("Unary - requires numeric operand (got " + t->str() + ")", pos);
        return t;
    }

    TypePtr validate_bin_op(BinOp op, Expr* lhs, Expr* rhs, Position pos) {
        TypePtr l = validate_expr(lhs);
        TypePtr r = validate_expr(rhs);
        // Special case: Matrix Multiplication
        if (op == BinOp::MatMul) {
            if (!l->is_infer() && l->kind != Type::Kind::Tensor)
                error("@ requires Tensor operands (got " + l->str() + ")", pos);
            if (!r->is_infer() && r->kind != Type::Kind::Tensor)
                error("@ requires Tensor operands (got " + r->str() + ")", pos);
            // Shape check: (M×N) @ (N×P) — inner dims must match
            if (l->kind == Type::Kind::Tensor && r->kind == Type::Kind::Tensor && !l->shape.empty() && !r->shape.empty()) {
                const Dim& l_inner = l->shape.back();
                const Dim& r_inner = r->shape.front();
                if (!dim_compat(l_inner, r_inner))
                    error("MatMul inner dimension mismatch: " +
                          dim_str(l_inner) + " != " + dim_str(r_inner), pos);
                // Build result shape: l->shape[0..-2] + r->shape[1..]
                std::vector<Dim> result_shape;
                for (size_t i = 0; i + 1 < l->shape.size(); ++i)
                    result_shape.push_back(l->shape[i]);
                for (size_t i = 1; i < r->shape.size(); ++i)
                    result_shape.push_back(r->shape[i]);
                return Type::tensor(l->elem_type(), std::move(result_shape));
            }
            return Type::tensor(l->elem_type());
        }
        if (op == BinOp::And || op == BinOp::Or)
        {
            expect_compat(Type::bool_(), l, pos, "Left operand of && / || must be bool");
            expect_compat(Type::bool_(), r, pos, "Right operand of && / || must be bool");
            return Type::bool_();
        }
        if (l->kind == Type::Kind::Tensor && r->kind != Type::Kind::Tensor) {
            expect_compat(l->elem_type(), r, pos,
                          "Tensor broadcast: scalar type must match element type");
            return l;   // result has same shape and elem type as the tensor
        }
        if (r->kind == Type::Kind::Tensor && l->kind != Type::Kind::Tensor) {
            expect_compat(r->elem_type(), l, pos,
                          "Tensor broadcast: scalar type must match element type");
            return r;
        }
        // Standard Arithmetic/Logical Compatibility
        expect_compat(l, r, pos, "Binary operator type mismatch");
        // Comparison operators return Bool
        if (op == BinOp::Eq || op == BinOp::Neq || op == BinOp::Lt || op == BinOp::Gt || op == BinOp::Lte|| op == BinOp::Gte) {
            return Type::bool_();
        }
        return l;
    }

    TypePtr validate_call(Expr* expr) {
        Expr* callee = expr->kind.callee.get();
        if (!callee || callee->kind.tag != ExprKind::Tag::Id)
            return Type::infer();
        const std::string& fn_name = callee->kind.id.name();
        Symbol* s = symb_tab.lookup(fn_name);
        if (!s) error("Calling undefined function '" + fn_name + "'", expr->pos);
        auto& args = expr->kind.args;
        if (s->type && s->type->kind == Type::Kind::Fn) {
            auto params = s->type->param_types();
            if (args.size() != params.size())
                error("Function '" + fn_name + "' expects " +
                    std::to_string(params.size()) +
                    " argument(s), got " + std::to_string(args.size()),
                    expr->pos);
            SubstMap subst;
            std::vector<TypePtr> actual_tys;
            actual_tys.reserve(args.size());
            for (size_t i = 0; i < args.size(); ++i) {
                TypePtr actual = validate_expr(args[i].get());
                actual_tys.push_back(actual);
                collect_subst(params[i], actual, subst, args[i]->pos);
            }
            for (size_t i = 0; i < args.size(); ++i) {
                TypePtr expected = apply_subst(params[i], subst);
                expect_compat(expected, actual_tys[i], args[i]->pos,
                              "Argument " + std::to_string(i + 1) +
                              " of '" + fn_name + "' type mismatch");
            }
            return apply_subst(s->type->ret_type(), subst);
        }
        for (auto& arg : args)
            validate_expr(arg.get());
        return s->type ? s->type : Type::infer();
    }

    TypePtr validate_pipe(Expr* expr) {
        TypePtr lhs_ty = validate_expr(expr->kind.pipe_lhs.get());
        Expr* rhs = expr->kind.pipe_rhs.get();
        if (!rhs) return Type::infer();
        if (rhs->kind.tag != ExprKind::Tag::Call) error("RHS of pipe operator must be a function call", expr->pos);
        Expr* rhs_callee = rhs->kind.callee.get();
        if (!rhs_callee || rhs_callee->kind.tag != ExprKind::Tag::Id) return Type::infer();
        const std::string& fn_name = rhs_callee->kind.id.name();
        Symbol* sym = symb_tab.lookup(fn_name);
        if (!sym) error("Pipe: undefined function '" + fn_name + "'", rhs->pos);
        if (!sym->type || sym->type->kind != Type::Kind::Fn)
            return Type::infer();
        auto params = sym->type->param_types();
        if (params.empty())
            error("Pipe: '" + fn_name + "' takes no arguments", rhs->pos);
        expect_compat(params[0], lhs_ty, expr->pos,
                      "Pipe: " + lhs_ty->str() +
                      " cannot flow into first parameter of '" + fn_name +
                      "' (" + params[0]->str() + ")");
        auto& provided = rhs->kind.args;
        if (provided.size() != params.size() - 1)
            error("Pipe: '" + fn_name + "' expects " +
                  std::to_string(params.size() - 1) +
                  " additional argument(s), got " +
                  std::to_string(provided.size()), rhs->pos);
        for (size_t i = 0; i < provided.size(); ++i) {
            TypePtr actual = validate_expr(provided[i].get());
            expect_compat(params[i + 1], actual, provided[i]->pos,
                          "Pipe: argument " + std::to_string(i + 2) +
                          " of '" + fn_name + "' type mismatch");
        }
        return sym->type->ret_type();
    }

    TypePtr validate_match(Stmt* stmt) {
        TypePtr subject_ty = validate_expr(stmt->kind.match_subject.get());
        for (auto& arm : stmt->kind.match_arms) {
            symb_tab.pushScope();
            bool is_wildcard = arm.pattern && arm.pattern->kind.tag == ExprKind::Tag::Id && arm.pattern->kind.id.name() == "_";
            if (!is_wildcard)
            {
                TypePtr pattern_ty = validate_expr(arm.pattern.get());
                expect_compat(subject_ty, pattern_ty, arm.pos, "Match pattern type mismatch");
            }
            // Validate guard
            if (arm.guard) {
                TypePtr guard_ty = validate_expr(arm.guard.value().get());
                expect_compat(Type::bool_(), guard_ty, arm.pos, "Match guard must be bool");
            }
            // Validate body (Statement version uses body_stmt)
            if (arm.hasStmtBody()) validate_stmt(arm.body_stmt.get());
            else                   validate_expr(arm.body.get());
            symb_tab.popScope();
        }
        return Type::void_();
    }

    // --- Helpers ---

    TypePtr validate_compound(Compound& comp, bool scope_already_pushed) {
        if (!scope_already_pushed) symb_tab.pushScope();
        TypePtr last_ty = Type::void_();
        for (auto& s : comp.stmts) {
            last_ty = validate_stmt(s.get());
        }
        if (comp.tail_expr) {
            last_ty = validate_expr(comp.tail_expr.get());
        }
        if (!scope_already_pushed) symb_tab.popScope();
        return last_ty;
    }

    void validate_structs()
    {
        // colour map — all start white (absent from map)
        std::unordered_map<std::string, int> colour;
        // DFS: returns the name that closes a cycle, or "" if clean.
        std::function<void(const std::string&, const std::string&)> dfs =
            [&](const std::string& name, const std::string& via)
        {
            auto it = struct_registry.find(name);
            if (it == struct_registry.end())
                error("Struct field references unknown type '" + name +
                      "' (via '" + via + "')", Position{});
            int& c = colour[name];
            if (c == 2) return;            // already fully checked
            if (c == 1)                    // back edge → cycle
                error("Struct '" + name +
                      "' is part of a value-cycle (through '" + via + "'). "
                      "Use an indirect type to break the cycle.", Position{});
            c = 1;   // grey — on stack
            for (auto& [field_name, field_ty] : it->second.fields) {
                if (!field_ty || field_ty->kind != Type::Kind::Named) continue;
                const std::string& dep = field_ty->type_name;
                dfs(dep, name + "." + field_name);
            }
            c = 2;   // black — done
        };
        for (auto& [name, _] : struct_registry)
            if (colour[name] == 0)
                dfs(name, name);
    }

    TypePtr validate_if(Stmt* stmt) {
        TypePtr cond = validate_expr(stmt->kind.if_cond.get());
        expect_compat(Type::bool_(), cond, stmt->pos, "if condition must be bool");
        validate_compound(stmt->kind.if_body, false);
        if (stmt->kind.else_or_else_if) validate_stmt(stmt->kind.else_or_else_if.get());
        return Type::void_();
    }

    TypePtr validate_spawn(Stmt* stmt) {
        if (!stmt->kind.spawn_fn) return Type::void_();
        int saved = iteration_depth;
        iteration_depth = 0;
        async_depth++;
        TypePtr inner_ty = validate_stmt(stmt->kind.spawn_fn.get());
        async_depth--;
        iteration_depth = saved;
        return Type::task(inner_ty ? inner_ty : Type::void_());
    }

    TypePtr validate_for(Stmt* stmt)
    {
        TypePtr iter_ty = validate_expr(stmt->kind.for_iter.get());
        TypePtr elem_ty = Type::infer();
        if (iter_ty) {
            switch (iter_ty->kind) {
                case Type::Kind::I32:
                case Type::Kind::I64:
                    elem_ty = Type::i32();  // range iterator
                    break;
                case Type::Kind::Array:
                case Type::Kind::Set:
                case Type::Kind::Queue:
                case Type::Kind::Stack:
                    // elem_type() returns the inner TypePtr (may be Infer)
                    elem_ty = iter_ty->elem_type();
                    break;
                case Type::Kind::Tensor:
                    elem_ty = Type::f32();  // scalar iteration over tensor
                    break;
                default:
                    elem_ty = Type::infer();
                    break;
            }
        }
        symb_tab.pushScope();
        iteration_depth++;
        Symbol loop_sym(stmt->kind.for_var, elem_ty, IdentCtx::Def, stmt->pos);
        loop_sym.is_mutable = false;   // for-loop variable cannot be reassigned
        loop_sym.requires_grad = false;  // loop counters never require gradients
        symb_tab.define(loop_sym);
        validate_compound(stmt->kind.for_body, /*scope_already_pushed=*/true);
        iteration_depth--;
        symb_tab.popScope();
        return Type::void_();
    }

    TypePtr validate_while(Stmt* stmt) {
        if (stmt->kind.while_cond) {
            TypePtr cond = validate_expr(stmt->kind.while_cond.get());
            expect_compat(Type::bool_(), cond, stmt->pos, "while condition must be bool");
        }
        iteration_depth++;
        validate_compound(stmt->kind.while_body, false);
        iteration_depth--;
        return Type::void_();
    }

    TypePtr validate_lit(const LitKind& lit) {
        switch (lit.tag) {
            case LitKind::Tag::Int:   return Type::i32();
            case LitKind::Tag::Float: return Type::f32();
            case LitKind::Tag::Str:   return Type::str_();
            case LitKind::Tag::Bool:  return Type::bool_();
            default: return Type::void_();
        }
    }

    void expect_compat(const TypePtr& expected, const TypePtr& actual, Position pos, const std::string& ctx) {
        if (!expected || !actual) return;
        if (expected->is_infer()) return;
        if (actual->is_infer()) return;
        if (expected->kind == Type::Kind::Var) return;
        if (actual->kind == Type::Kind::Var) return;
        if (!type_compat(expected, actual))
            error(ctx + " (expected " + expected->str() + ", got " + actual->str() + ")", pos);
        }

    using SubstMap = std::unordered_map<std::string, TypePtr>;
    void collect_subst(const TypePtr& param, const TypePtr& actual,
                       SubstMap& subst, Position pos)
    {
        if (!param || !actual) return;
        if (param->is_infer() || actual->is_infer()) return;
        if (param->kind == Type::Kind::Var) {
            const std::string& tvar = param->type_name;
            auto it = subst.find(tvar);
            if (it == subst.end()) {
                subst[tvar] = actual;   // first occurrence — bind it
            } else {
                // Subsequent occurrence — must match the already-bound type
                expect_compat(it->second, actual, pos,
                              "Generic type parameter '" + tvar +
                              "' inferred as inconsistent types (" +
                              it->second->str() + " vs " + actual->str() + ")");
            }
            return;
        }
        // Recurse into composite types (Array<T>, Fn(T)->U, etc.)
        if (param->kind == actual->kind &&
            param->args.size() == actual->args.size())
        {
            for (size_t i = 0; i < param->args.size(); ++i)
                collect_subst(param->args[i], actual->args[i], subst, pos);
        }
    }

    TypePtr apply_subst(const TypePtr& ty, const SubstMap& subst)
    {
        if (!ty) return ty;
        if (ty->kind == Type::Kind::Var) {
            auto it = subst.find(ty->type_name);
            return (it != subst.end()) ? it->second : ty;
        }
        if (ty->args.empty()) return ty;   // primitive or Named — no vars inside
        // Rebuild with substituted children
        auto result = std::make_shared<Type>(ty->kind);
        result->type_name = ty->type_name;
        result->shape     = ty->shape;
        for (auto& a : ty->args)
            result->args.push_back(apply_subst(a, subst));
        return result;
    }

    static TyKind tkFromType(const TypePtr& t) {
        if (!t) return TyKind::Infer;
        switch (t->kind) {
            case Type::Kind::I32:    return TyKind::I32;
            case Type::Kind::I64:    return TyKind::I64;
            case Type::Kind::F32:    return TyKind::F32;
            case Type::Kind::F64:    return TyKind::F64;
            case Type::Kind::Bool:   return TyKind::Bool;
            case Type::Kind::Str:    return TyKind::Str;
            case Type::Kind::Void:   return TyKind::Void;
            case Type::Kind::Tensor: return TyKind::Tensor;
            case Type::Kind::Array:  return TyKind::Array;
            case Type::Kind::Map:    return TyKind::Map;
            case Type::Kind::Set:    return TyKind::Set;
            case Type::Kind::Queue:  return TyKind::Queue;
            case Type::Kind::Stack:  return TyKind::Stack;
            case Type::Kind::Tuple:  return TyKind::Tuple;
            case Type::Kind::Fn:     return TyKind::FnType;
            case Type::Kind::Named:  return TyKind::UserDef;
            case Type::Kind::Var:    return TyKind::Generic;
            case Type::Kind::Infer:  return TyKind::Infer;
            default:                 return TyKind::Infer;
        }
    }

    [[noreturn]]
    void error(const std::string& msg, Position pos) {
        throw std::runtime_error("[" + std::to_string(pos.line) + ":" + 
            std::to_string(pos.column) + "] " + msg);
    }
};