#pragma once

#include "ASTNode.h"
#include "SymbolTable.h"
#include <stdexcept>
#include <string>
#include <iostream>

struct FieldType {
    TyKind kind;
    std::string user_name;
};

struct StructDef {
    std::string name;
    std::unordered_map<std::string, FieldType> fields;
    
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
    TyKind expected_ret_ty = TyKind::Void;
    int iteration_depth = 0;

    void register_struct(const StructDef& def)
    {
        struct_registry[def.name] = def;
    }

    void register_top_level(Stmt* stmt)
    {
        if (stmt->kind.tag == StmtKind::Tag::Struct)
        {
            StructDef def;
            def.name = stmt->kind.struct_name;
            for (auto& f : stmt->kind.struct_fields)
                def.fields[f.name] = FieldType{ f.ty, f.user_type_name };
            register_struct(def);
        }
        if (stmt->kind.tag == StmtKind::Tag::Func)
        {
            auto& func = stmt->kind.func.value();
            std::vector<TyKind> p_tys;
            for (auto& p : func.params)
                p_tys.push_back(p.ty_kind());
            symb_tab.define(
                Symbol(func.ident.name(),
                    func.ident.ty_kind(),
                    IdentCtx::FuncDef,
                    func.ident.pos(),
                    {},
                    p_tys));
        }
    }

    // --- Statement Validation ---
    TyKind validate_stmt(Stmt* stmt) {
        if (!stmt) return TyKind::Void;
        switch (stmt->kind.tag) {
            case StmtKind::Tag::Let: return validate_let(stmt);
            case StmtKind::Tag::Func: {
                if (stmt->kind.func.has_value()) {
                    return validate_func(stmt->kind.func.value());
                }
                return TyKind::Void;
            }
            case StmtKind::Tag::Return: {
                TyKind actual = stmt->kind.ret_expr
                    ? validate_expr(stmt->kind.ret_expr.get())
                    : TyKind::Void;
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
                return TyKind::Void;
            }
            case StmtKind::Tag::Match: return validate_match(stmt);
            case StmtKind::Tag::Spawn: return validate_spawn(stmt);
            case StmtKind::Tag::Import: return TyKind::Void; // TODO: open registry
            default: return TyKind::Void;
        }
        return TyKind::Void; 
    }

    TyKind validate_let(Stmt* stmt) { // let x [: T] = expr
        // Validate RHS first to prevent recursive definitions like 'let x = x'
        TyKind rhs_ty = validate_expr(stmt->kind.let_expr.get());
        TyKind var_ty = stmt->kind.let_ident.ty_kind();
        if (var_ty == TyKind::Infer) {
            var_ty = rhs_ty;
            stmt->kind.let_ident.set_ty_kind(rhs_ty);
        } else {
            expect_compat(var_ty, rhs_ty, stmt->pos, "Variable assignment type mismatch");
        }
        // Extract shape from generic params if present, Define in Symbol Table
        std::vector<int> shape;
        TyKind elem_ty = TyKind::Infer;
        if (var_ty == TyKind::Tensor && stmt->kind.let_expr && stmt->kind.let_expr->kind.generic_params.has_value())
        {
            shape = stmt->kind.let_expr->kind.generic_params->shape;
        }
        if (var_ty == TyKind::Array && stmt->kind.let_expr) {
            auto& elems = stmt->kind.let_expr->kind.elements;
            if (!elems.empty()) {
                if (elems[0]->kind.tag == ExprKind::Tag::Lit) {
                    switch (elems[0]->kind.lit.tag) {
                        case LitKind::Tag::Int:   elem_ty = TyKind::I32;  break;
                        case LitKind::Tag::Float: elem_ty = TyKind::F32;  break;
                        case LitKind::Tag::Bool:  elem_ty = TyKind::Bool; break;
                        case LitKind::Tag::Str:   elem_ty = TyKind::Str;  break;
                        default:                  elem_ty = TyKind::Infer; break;
                    }
                } else {
                    // Non-literal first element (Id, Call, Binary…) — resolve normally.
                    elem_ty = validate_expr(elems[0].get());
                }
            }
        }
        Symbol sym(stmt->kind.let_ident.name(), var_ty, IdentCtx::Def, stmt->pos, shape);
        if (var_ty == TyKind::UserDef)
            sym.user_type_name = stmt->kind.let_ident.type_name();
        sym.elem_ty = elem_ty;
        symb_tab.define(sym);
        return var_ty;
    }

    TyKind validate_func(Func& func) {
        std::vector<TyKind> p_tys;
        for (auto& p : func.params) {
            p_tys.push_back(p.ty_kind());
        }
        // Register function name in current scope
        // symb_tab.define(Symbol(func.ident.name(), func.ident.ty_kind(), IdentCtx::FuncDef, func.ident.pos(), {}, p_tys));
        if (!symb_tab.existsInCurrentScope(func.ident.name()))
            symb_tab.define(Symbol(func.ident.name(), func.ident.ty_kind(), IdentCtx::FuncDef, func.ident.pos(), {}, p_tys));
        symb_tab.pushScope();
        TyKind old_expect = expected_ret_ty;
        expected_ret_ty = func.ident.ty_kind();
        // Register parameters in function scope
        for (auto& param : func.params) {
            symb_tab.define(Symbol(param.name(), param.ty_kind(), IdentCtx::Param, param.pos()));
        }
        TyKind body_ret_ty = validate_compound(func.body, true);
        // Ensure return type matches
        if (func.ident.ty_kind() != TyKind::Void)
            expect_compat(func.ident.ty_kind(), body_ret_ty, func.ident.pos(), "Function '" + func.ident.name() + "' return type mismatch");
        expected_ret_ty = old_expect;
        symb_tab.popScope();
        return func.ident.ty_kind();
    }

    // --- Expression Validation ---

    TyKind validate_expr(Expr* expr) {
        if (!expr) return TyKind::Void;
        switch (expr->kind.tag) {
            case ExprKind::Tag::Lit: return validate_lit(expr->kind.lit);
            case ExprKind::Tag::Id: {
                const std::string& name = expr->kind.id.name();
                if (name == "_") return TyKind::Infer;
                Symbol* s = symb_tab.lookup(name);
                if (!s) error("Undefined identifier: " + expr->kind.id.name(), expr->pos);
                // Update AST node with the resolved type for the code generator
                expr->kind.id.set_ty_kind(s->type);
                if (s->type == TyKind::UserDef) {
                    expr->kind.resolved_user_type = s->user_type_name;
                }
                return s->type;
            }
            case ExprKind::Tag::Unary: return validate_unary(expr->kind.unary_op, expr->kind.operand.get(), expr->pos);
            case ExprKind::Tag::Binary: return validate_bin_op(expr->kind.bin_op, expr->kind.lhs.get(), expr->kind.rhs.get(), expr->pos);
            case ExprKind::Tag::Assign: {
                TyKind l = validate_expr(expr->kind.lhs.get());
                TyKind r = validate_expr(expr->kind.rhs.get());
                expect_compat(l, r, expr->pos, "Assignment type mismatch");
                return TyKind::Void;
            }
            case ExprKind::Tag::Call: return validate_call(expr);
            case ExprKind::Tag::Pipe: return validate_pipe(expr);
            case ExprKind::Tag::Grad: { // grad(loss, params)
                TyKind loss_ty = validate_expr(expr->kind.grad_loss.get());
                if (loss_ty != TyKind::F32 && loss_ty != TyKind::F64 && loss_ty != TyKind::Infer) {
                    error("grad() requires a scalar loss (F32 or F64)", expr->pos);
                }
                TyKind param_ty = validate_expr(expr->kind.grad_params.get());
                if (param_ty != TyKind::Tensor && param_ty != TyKind::Infer) {
                    error("Can only compute gradients with respect to Tensors", expr->pos);
                }
                return param_ty; 
            }
            case ExprKind::Tag::If: {
                TyKind cond_ty = validate_expr(expr->kind.condition.get());
                expect_compat(TyKind::Bool, cond_ty, expr->pos, "if condition must be bool");
                TyKind then_ty = validate_expr(expr->kind.then_branch.get());
                if (expr->kind.else_branch)
                {
                    TyKind else_ty = validate_expr(expr->kind.else_branch.get());
                    expect_compat(then_ty, else_ty, expr->pos, "if/else branches must have the same type");
                }
                return then_ty;
            }
            case ExprKind::Tag::Block: return validate_compound(expr->kind.block, false);
            case ExprKind::Tag::TensorLit: return TyKind::Tensor;
            case ExprKind::Tag::ArrayLit: {
                if (expr->kind.elements.empty()) return TyKind::Array;
                TyKind first_ty = validate_expr(expr->kind.elements[0].get());
                for (size_t i = 1; i < expr->kind.elements.size(); ++i) {
                    TyKind current_ty = validate_expr(expr->kind.elements[i].get());
                    expect_compat(first_ty, current_ty, expr->kind.elements[i]->pos, "Array elements must have consistent types");
                }
                return TyKind::Array;
            }
            case ExprKind::Tag::SetLit:
            case ExprKind::Tag::QueueLit:
            case ExprKind::Tag::StackLit:
            case ExprKind::Tag::TupleLit: {
                for (auto& el : expr->kind.elements) {
                    validate_expr(el.get());
                }
                if (expr->kind.tag == ExprKind::Tag::SetLit) return TyKind::Set;
                if (expr->kind.tag == ExprKind::Tag::QueueLit) return TyKind::Queue;
                if (expr->kind.tag == ExprKind::Tag::StackLit) return TyKind::Stack;
                return TyKind::Tuple;
            }
            case ExprKind::Tag::MapLit: {
                if (expr->kind.map_pairs.empty()) return TyKind::Map;
                // Map{"key": value} -> map_pairs is vector<pair<ExprPtr, ExprPtr>>
                TyKind k_ty = validate_expr(expr->kind.map_pairs[0].first.get());
                TyKind v_ty = validate_expr(expr->kind.map_pairs[0].second.get());
                for (size_t i = 1; i < expr->kind.map_pairs.size(); ++i) {
                    auto& pair = expr->kind.map_pairs[i];
                    expect_compat(k_ty, validate_expr(pair.first.get()), expr->pos, "Inconsistent Map key");
                    expect_compat(v_ty, validate_expr(pair.second.get()), expr->pos, "Inconsistent Map value");
                }
                return TyKind::Map;
            }
            case ExprKind::Tag::Index: { // expr[i]
                TyKind obj = validate_expr(expr->kind.target.get());
                TyKind idx = validate_expr(expr->kind.index.get());
                if (idx != TyKind::I32 && idx != TyKind::I64 && idx != TyKind::Infer) error("Index subscript must be integer", expr->pos);
                if (obj != TyKind::Tensor) return TyKind::Infer;
                if (expr->kind.target->kind.tag == ExprKind::Tag::Id) {
                    Symbol* s = symb_tab.lookup(expr->kind.target->kind.id.name());
                    if (s && !s->shape.empty()) return (s->shape.size() == 1) ? TyKind::F32 : TyKind::Tensor;
                }
                return TyKind::F32;
            }
            case ExprKind::Tag::Field: { // expr.member
                TyKind target_ty = validate_expr(expr->kind.target.get());
                std::string struct_name;
                if (expr->kind.target->kind.tag == ExprKind::Tag::Id)
                {
                    const std::string& var_name = expr->kind.target->kind.id.name();
                    Symbol* s = symb_tab.lookup(var_name);
                    if (!s) error("Undefined identifier '" + var_name + "'", expr->pos);
                    // Non-UserDef dot access (e.g. future tensor.shape) — stub.
                    if (s->type != TyKind::UserDef) return TyKind::Infer;
                    else if (target_ty == TyKind::UserDef) {
                        struct_name = expr->kind.target->kind.resolved_user_type;
                        return TyKind::Infer; 
                    } else return TyKind::Infer; 
                    // UserDef without a registered struct — registry not yet populated.
                    if (struct_name.empty()) return TyKind::Infer;
                    auto it = struct_registry.find(struct_name);
                    if (it == struct_registry.end()) error("Unknown struct type '" + struct_name + "'", expr->pos);
                    auto fit = it->second.fields.find(expr->kind.member);
                    if (fit == it->second.fields.end()) error("Struct '" + struct_name + "' has no field '" + expr->kind.member + "'", expr->pos);
                    expr->kind.resolved_user_type = fit->second.user_name;
                    return fit->second.kind;
                }
                return TyKind::Infer;
            }
            case ExprKind::Tag::Scope: { // expr::member
                validate_expr(expr->kind.target.get());
                return TyKind::Infer; // TODO: ImportRegistry
            }
            case ExprKind::Tag::Range: { // lo..hi
                TyKind lo = validate_expr(expr->kind.lhs.get());
                TyKind hi = validate_expr(expr->kind.rhs.get());
                expect_compat(lo, hi, expr->pos, "Range bounds must match");
                return lo;
            }
            case ExprKind::Tag::ChannelSend: { // ch <- val
                validate_expr(expr->kind.channel.get());
                validate_expr(expr->kind.send_val.get());
                return TyKind::Void;
            }
            case ExprKind::Tag::Await: return validate_expr(expr->kind.awaited.get());
            case ExprKind::Tag::FnExpr: { // fn(x:T)->T { }
                symb_tab.pushScope();
                TyKind saved_ret = expected_ret_ty;
                expected_ret_ty  = expr->kind.fn_ret_type;
                for (auto& param : expr->kind.fn_params) {
                    // fn_params is vector<pair<string, TyKind>>
                    symb_tab.define(Symbol(param.first, param.second, IdentCtx::Param, expr->pos));
                }
                TyKind body_ty = validate_compound(expr->kind.fn_body, true);
                if (expr->kind.fn_ret_type != TyKind::Void) expect_compat(expr->kind.fn_ret_type, body_ty, expr->pos, "Lambda return type mismatch");
                expected_ret_ty = saved_ret;
                symb_tab.popScope();
                return TyKind::FnType;
            }
            case ExprKind::Tag::Match: { // similar to validate_match function
                TyKind subject_ty = validate_expr(expr->kind.match_subject.get());
                TyKind match_ret_ty = TyKind::Infer;
                bool first_arm = true;
                for (auto& arm : expr->kind.arms) {
                    symb_tab.pushScope();
                    bool is_wildcard = arm.pattern && arm.pattern->kind.tag == ExprKind::Tag::Id && arm.pattern->kind.id.name() == "_";
                    if (!is_wildcard) {
                        TyKind pattern_ty = validate_expr(arm.pattern.get());
                        expect_compat(subject_ty, pattern_ty, arm.pos, "Match pattern type mismatch");
                    }
                    if (arm.guard) {
                        TyKind guard_ty = validate_expr(arm.guard.value().get());
                        expect_compat(TyKind::Bool, guard_ty, arm.pos, "Match guard must be bool");
                    }
                    TyKind arm_ty = arm.hasStmtBody() ? validate_stmt(arm.body_stmt.get()) : validate_expr(arm.body.get());
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
                return TyKind::Void;
        }
    }

    TyKind validate_unary(UnaryOp op, Expr* operand, Position pos)
    {
        TyKind t = validate_expr(operand);
        if (op == UnaryOp::Not)
        {
            expect_compat(TyKind::Bool, t, pos,
                          "! operator requires bool operand");
            return TyKind::Bool;
        }
        // UnaryOp::Neg — valid for numeric types only.
        if (t != TyKind::I32 && t != TyKind::I64 &&
            t != TyKind::F32 && t != TyKind::F64 &&
            t != TyKind::Infer)
            error("Unary - requires numeric operand (got " +
                  tyName(t) + ")", pos);
        return t;
    }

    TyKind validate_bin_op(BinOp op, Expr* lhs, Expr* rhs, Position pos) {
        TyKind l = validate_expr(lhs);
        TyKind r = validate_expr(rhs);
        // Special case: Matrix Multiplication
        if (op == BinOp::MatMul) {
            if (l != TyKind::Tensor || r != TyKind::Tensor) {
                error("MatMul (@) is only defined for Tensors", pos);
            }
            if (lhs->kind.tag == ExprKind::Tag::Id && rhs->kind.tag == ExprKind::Tag::Id) {
                Symbol* sL = symb_tab.lookup(lhs->kind.id.name());
                Symbol* sR = symb_tab.lookup(rhs->kind.id.name());
                if (sL && sR && !sL->shape.empty() && !sR->shape.empty()) {
                    // Rule: (M x N) @ (N x P) -> (M x P)
                    if (sL->shape.back() != sR->shape[0]) {
                        error("Incompatible tensor shapes for MatMul: " + 
                            std::to_string(sL->shape.back()) + " != " + 
                            std::to_string(sR->shape[0]), pos);
                    }
                }
            }
            return TyKind::Tensor;
        }
        if (op == BinOp::And || op == BinOp::Or)
        {
            expect_compat(TyKind::Bool, l, pos, "Left operand of && / || must be bool");
            expect_compat(TyKind::Bool, r, pos, "Right operand of && / || must be bool");
            return TyKind::Bool;
        }
        // Standard Arithmetic/Logical Compatibility
        expect_compat(l, r, pos, "Binary operator type mismatch");
        // Comparison operators return Bool
        if (op == BinOp::Eq || op == BinOp::Neq || op == BinOp::Lt || op == BinOp::Gt || op == BinOp::Lte|| op == BinOp::Gte) {
            return TyKind::Bool;
        }
        return l;
    }

    TyKind validate_call(Expr* expr) {
        Expr* callee = expr->kind.callee.get();
        if (!callee || callee->kind.tag != ExprKind::Tag::Id)
            return TyKind::Infer;
        const std::string& fn_name = callee->kind.id.name();
        Symbol* s = symb_tab.lookup(fn_name);
        if (!s) error("Calling undefined function '" + fn_name + "'", expr->pos);
        auto& args = expr->kind.args;
        if (args.size() != s->param_types.size()) {
            error("Argument count mismatch. Expected " + std::to_string(s->param_types.size()), expr->pos);
        }
        for (size_t i = 0; i < args.size(); ++i) {
            TyKind actual = validate_expr(args[i].get());
            expect_compat(s->param_types[i], actual, args[i]->pos, "Argument type mismatch");
        }
        return s->type;
    }

    TyKind validate_pipe(Expr* expr) {
        TyKind lhs_ty = validate_expr(expr->kind.pipe_lhs.get());
        Expr* rhs = expr->kind.pipe_rhs.get();
        if (!rhs) return TyKind::Infer;
        if (rhs->kind.tag != ExprKind::Tag::Call) error("RHS of pipe operator must be a function call", expr->pos);
        Expr* rhs_callee = rhs->kind.callee.get();
        if (!rhs_callee || rhs_callee->kind.tag != ExprKind::Tag::Id) return TyKind::Infer;
        const std::string& fn_name = rhs_callee->kind.id.name();
        Symbol* sym = symb_tab.lookup(fn_name);
        if (!sym) error("Pipe: undefined function '" + fn_name + "'", rhs->pos);
        if (sym->param_types.empty()) error("Pipe: function '" + fn_name + "' takes no arguments and cannot be piped into", rhs->pos);
        // First parameter receives the lhs value.
        expect_compat(sym->param_types[0], lhs_ty, expr->pos,
            "Pipe type mismatch: " + tyName(lhs_ty) +
            " cannot flow into first parameter of '" +
            fn_name + "' (" + tyName(sym->param_types[0]) + ")");
        // Remaining explicit args cover parameters [1..N].
        auto& provided = rhs->kind.args;
        size_t expected_extra = sym->param_types.size() - 1;
        if (provided.size() != expected_extra)
            error("Pipe: '" + fn_name + "' expects " +
                std::to_string(expected_extra) +
                " additional argument(s), got " +
                std::to_string(provided.size()), rhs->pos);
        for (size_t i = 0; i < provided.size(); ++i)
        {
            TyKind actual = validate_expr(provided[i].get());
            expect_compat(sym->param_types[i + 1], actual,
                provided[i]->pos,
                "Pipe: argument " + std::to_string(i + 2) +
                " of '" + fn_name + "' type mismatch");
        }
        return sym->type;
    }

    TyKind validate_match(Stmt* stmt) {
        TyKind subject_ty = validate_expr(stmt->kind.match_subject.get());
        for (auto& arm : stmt->kind.match_arms) {
            symb_tab.pushScope();
            bool is_wildcard = arm.pattern && arm.pattern->kind.tag == ExprKind::Tag::Id && arm.pattern->kind.id.name() == "_";
            if (!is_wildcard)
            {
                TyKind pattern_ty = validate_expr(arm.pattern.get());
                expect_compat(subject_ty, pattern_ty, arm.pos, "Match pattern type mismatch");
            }
            // Validate guard
            if (arm.guard) {
                TyKind guard_ty = validate_expr(arm.guard.value().get());
                expect_compat(TyKind::Bool, guard_ty, arm.pos, "Match guard must be bool");
            }
            // Validate body (Statement version uses body_stmt)
            if (arm.hasStmtBody()) validate_stmt(arm.body_stmt.get());
            else                   validate_expr(arm.body.get());
            symb_tab.popScope();
        }
        return TyKind::Void;
    }

    // --- Helpers ---

    TyKind validate_compound(Compound& comp, bool scope_already_pushed) {
        if (!scope_already_pushed) symb_tab.pushScope();
        TyKind last_ty = TyKind::Void;
        for (auto& s : comp.stmts) {
            last_ty = validate_stmt(s.get());
        }
        if (comp.tail_expr) {
            last_ty = validate_expr(comp.tail_expr.get());
        }
        if (!scope_already_pushed) symb_tab.popScope();
        return last_ty;
    }

    TyKind validate_if(Stmt* stmt) {
        TyKind cond = validate_expr(stmt->kind.if_cond.get());
        expect_compat(TyKind::Bool, cond, stmt->pos, "if condition must be bool");
        validate_compound(stmt->kind.if_body, false);
        if (stmt->kind.else_or_else_if) validate_stmt(stmt->kind.else_or_else_if.get());
        return TyKind::Void;
    }

    TyKind validate_spawn(Stmt* stmt) {
        if (!stmt->kind.spawn_fn) return TyKind::Void;
        int saved = iteration_depth;
        iteration_depth = 0;
        validate_stmt(stmt->kind.spawn_fn.get());
        iteration_depth = saved;
        return TyKind::Void;
    }

    TyKind validate_for(Stmt* stmt)
    {
        TyKind iter_ty = validate_expr(stmt->kind.for_iter.get());
        TyKind elem_ty = TyKind::Infer;
        switch (iter_ty) {
            case TyKind::I32:
            case TyKind::I64:
                elem_ty = TyKind::I32;
                break;
            case TyKind::Array:
                if (stmt->kind.for_iter && stmt->kind.for_iter->kind.tag == ExprKind::Tag::Id) {
                    Symbol* s = symb_tab.lookup(stmt->kind.for_iter->kind.id.name());
                    if (s && s->elem_ty != TyKind::Infer)
                        elem_ty = s->elem_ty;
                }
                break;
            case TyKind::Tensor:
                elem_ty = TyKind::F32;   // scalar iteration over a tensor
                break;
            default:
                elem_ty = TyKind::Infer; // unknown iterator; resolve later
                break;
        }
        symb_tab.pushScope();
        iteration_depth++;
        symb_tab.define(Symbol(stmt->kind.for_var, elem_ty, IdentCtx::Def, stmt->pos));
        validate_compound(stmt->kind.for_body, /*scope_already_pushed=*/true);
        iteration_depth--;
        symb_tab.popScope();
        return TyKind::Void;
    }

    TyKind validate_while(Stmt* stmt) {
        if (stmt->kind.while_cond) {
            TyKind cond = validate_expr(stmt->kind.while_cond.get());
            expect_compat(TyKind::Bool, cond, stmt->pos, "while condition must be bool");
        }
        iteration_depth++;
        validate_compound(stmt->kind.while_body, false);
        iteration_depth--;
        return TyKind::Void;
    }

    TyKind validate_lit(const LitKind& lit) {
        switch (lit.tag) {
            case LitKind::Tag::Int:   return TyKind::I32;
            case LitKind::Tag::Float: return TyKind::F32;
            case LitKind::Tag::Str:   return TyKind::Str;
            case LitKind::Tag::Bool:  return TyKind::Bool;
            default: return TyKind::Void;
        }
    }

    void expect_compat(TyKind expected, TyKind actual, Position pos, const std::string& ctx) {
        if (expected == TyKind::Infer || actual == TyKind::Infer) return;
        if (expected == TyKind::Generic || actual == TyKind::Generic) return;
        if (expected != actual) {
            error(ctx + " (expected " + tyName(expected) + ", got " + tyName(actual) + ")", pos);
        }
    }

    static std::string tyName(TyKind t)
    {
        switch (t)
        {
            case TyKind::I32:     return "i32";
            case TyKind::I64:     return "i64";
            case TyKind::F32:     return "f32";
            case TyKind::F64:     return "f64";
            case TyKind::Bool:    return "bool";
            case TyKind::Str:     return "str";
            case TyKind::Void:    return "void";
            case TyKind::Tensor:  return "Tensor";
            case TyKind::Map:     return "Map";
            case TyKind::Set:     return "Set";
            case TyKind::Queue:   return "Queue";
            case TyKind::Stack:   return "Stack";
            case TyKind::Tuple:   return "Tuple";
            case TyKind::Array:   return "Array";
            case TyKind::FnType:  return "fn";
            case TyKind::Generic: return "Generic";
            case TyKind::Infer:   return "<infer>";
            case TyKind::UserDef: return "UserDef";
            default:              return "?";
        }
    }

    [[noreturn]]
    void error(const std::string& msg, Position pos) {
        throw std::runtime_error("[" + std::to_string(pos.line) + ":" + 
            std::to_string(pos.column) + "] " + msg);
    }
};