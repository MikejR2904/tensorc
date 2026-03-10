#pragma once

#include "ASTNode.h"
#include "Type.h"
#include "SymbolTable.h"
#include <stdexcept>
#include <string>
#include <iostream>

struct StructDef {
    std::string name;
    std::unordered_map<std::string, TypePtr> fields;
    
};

class SemanticAnalyzer {
public:
    SemanticAnalyzer() : symb_tab() {}
    void validate(Program& program) {
        try {
            for (auto& stmt : program.stmts)
                register_top_level(stmt.get());
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
    std::unordered_map<std::string, StructDef> struct_registry;
    TypePtr expected_ret_ty = Type::void_();
    int iteration_depth = 0;

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
                TypePtr field_ty = Type::fromTyKind(f.ty, f.user_type_name, "", nullptr, {});
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
            case StmtKind::Tag::Import: return Type::void_(); // TODO: open registry
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
            var_ty = Type::fromTyKind(ann_k, stmt->kind.let_ident.user_type_name(), "", (ann_k == TyKind::Array  ||
                ann_k == TyKind::Tensor ||
                ann_k == TyKind::Set    ||
                ann_k == TyKind::Queue  ||
                ann_k == TyKind::Stack) ? rhs_ty : nullptr);
            expect_compat(var_ty, rhs_ty, stmt->pos, "Variable '" + stmt->kind.let_ident.name() + "' type mismatch");
        }
        symb_tab.define(Symbol(stmt->kind.let_ident.name(), var_ty, IdentCtx::Def, stmt->pos));
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
        // Register parameters in function scope
        for (auto& param : func.params) {
            symb_tab.define(Symbol(param.name(), Type::fromTyKind(param.ty_kind(), param.user_type_name()), IdentCtx::Param, param.pos()));
        }
        TypePtr body_ret_ty = validate_compound(func.body, true);
        // Ensure return type matches
        if (!ret->is_void())
            expect_compat(ret, body_ret_ty, func.ident.pos(), "Function '" + func.ident.name() + "' return type mismatch");
        expected_ret_ty = old_expect;
        symb_tab.popScope();
        return fn_ty;
    }

    // --- Expression Validation ---

    TypePtr validate_expr(Expr* expr) {
        if (!expr) return Type::void_();
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
                std::vector<int> shape;
                if (expr->kind.generic_params.has_value()) {
                    auto& gp = *expr->kind.generic_params;
                    if (!gp.type_params.empty())
                        elem = Type::fromTyKind(gp.type_params[0]);
                    shape = gp.shape;
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
                if (obj_ty->kind == Type::Kind::Named) {
                    const std::string& sname = obj_ty->type_name;
                    auto it = struct_registry.find(sname);
                    if (it == struct_registry.end())
                        error("Unknown struct type '" + sname + "'", expr->pos);
                    auto fit = it->second.fields.find(expr->kind.member);
                    if (fit == it->second.fields.end())
                        error("Struct '" + sname + "' has no field '" +
                              expr->kind.member + "'", expr->pos);
                    return fit->second;
                }
                return Type::infer();
            }
            case ExprKind::Tag::Scope: { // expr::member
                validate_expr(expr->kind.target.get());
                return Type::infer(); // TODO: ImportRegistry
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
            case ExprKind::Tag::Await: return validate_expr(expr->kind.awaited.get());
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
            if (l->kind == Type::Kind::Tensor && r->kind == Type::Kind::Tensor &&
                !l->shape.empty() && !r->shape.empty())
                if (l->shape.back() != r->shape[0])
                    error("MatMul shape mismatch: " +
                          std::to_string(l->shape.back()) + " != " +
                          std::to_string(r->shape[0]), pos);
            return Type::tensor(l->elem_type());
        }
        if (op == BinOp::And || op == BinOp::Or)
        {
            expect_compat(Type::bool_(), l, pos, "Left operand of && / || must be bool");
            expect_compat(Type::bool_(), r, pos, "Right operand of && / || must be bool");
            return Type::bool_();
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
            for (size_t i = 0; i < args.size(); ++i) {
                TypePtr actual = validate_expr(args[i].get());
                expect_compat(params[i], actual, args[i]->pos,
                              "Argument " + std::to_string(i + 1) +
                              " of '" + fn_name + "' type mismatch");
            }
            return s->type->ret_type();
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
        validate_stmt(stmt->kind.spawn_fn.get());
        iteration_depth = saved;
        return Type::void_();
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
        symb_tab.define(Symbol(stmt->kind.for_var, elem_ty, IdentCtx::Def, stmt->pos));
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