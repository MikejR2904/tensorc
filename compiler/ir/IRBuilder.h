#pragma once
 
/// ir/IRBuilder.h  — AST structs fully defined before IRBuilder uses them.
 
#include "IRModule.h"
#include "IRPrinter.h"
#include "../ast/ASTNode.h"
 
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
 
namespace ir {
 
using Scope = std::unordered_map<std::string, Value*>;
 
class IRBuilder
{
public:
    IRBuilder() = default;
 
    // ── Entry point ───────────────────────────────────────────────────────────
    void build(const Program& prog, IRModule* mod)
    {
        this->mod_ = mod;
        // for (auto& s : prog.stmts) {
        //     if (s) lower_top_level(*s);
        // }
        for (auto& stmt : prog.stmts) {
            if (stmt->kind.tag == StmtKind::Tag::Let) {
                lower_top_level(*stmt);
            } 
            
            if (stmt->kind.tag == StmtKind::Tag::Func && stmt->kind.func.has_value()) {
                auto& func = stmt->kind.func.value();
                // Peek inside the function body for Let statements to hoist
                for (auto& param : func.params) {
                    if (param.info.ty_kind == TyKind::Tensor && param.info.tensor_gp.has_value()) {
                        for (auto& dim : param.info.tensor_gp->shape) {
                            // If 'N' is unknown, define it as a symbolic IR constant
                            if (const std::string* dim_name = std::get_if<std::string>(&dim)) {
                                if (!lookup(*dim_name)) {
                                    // Inject into IR symbol table as a symbolic constant
                                    auto i32_ty = std::make_shared<Type>(Type::Kind::I32);
                                    auto sym_val = std::make_unique<ir::ConstantInt>(0, std::move(i32_ty)); 
                                    sym_val->name = "@" + *dim_name; // Name it so the IR knows it's 'N'
                                    ir::Value* val_ptr = sym_val.get();
                                    mod_->add_global(std::move(sym_val));
                                    define(*dim_name, val_ptr);
                                }
                            }
                        }
                    }
                }
                // 1b. Harvest nested 'let' from body (e.g. inside main)
                for (auto& body_stmt : func.body.stmts) {
                    if (body_stmt->kind.tag == StmtKind::Tag::Let) {
                        lower_top_level(*body_stmt);
                    }
                }
            }
        }
        for (auto& stmt : prog.stmts) {
            if (stmt->kind.tag == StmtKind::Tag::Func) {
                register_prototype(*stmt);
            }
        }

        // --- PASS 3: Definition ---
        for (auto& stmt : prog.stmts) {
            if (stmt->kind.tag == StmtKind::Tag::Func) {
                lower_func_body(*stmt);
            }
        }
    }

    TypePtr lower_type(const Ident& id) {
        return Type::fromTyKind(id.info.ty_kind);
    }

    void register_prototype(const Stmt& s) {
        auto& func = s.kind.func.value();
        std::string fn_name = func.ident.name();
        TypePtr ret_ty = lower_type(func.ident);
        std::vector<TypePtr> ptypes;
        for (auto& p : func.params) ptypes.push_back(lower_type(p));
        TypePtr fn_sig_type = Type::fn(ptypes, ret_ty);
        ir::Function* func_ptr = mod_->add_function(fn_name, fn_sig_type, false);
        global_functions[fn_name] = func_ptr;
        if (!func_ptr) return;

        // Register the function in the IR Module
        if (func_ptr->params.empty()) {
            for (size_t i = 0; i < func.params.size(); ++i) {
                auto& p = func.params[i];
                auto arg = std::make_shared<ir::Argument>(p.name(), lower_type(p), i);
                func_ptr->params.push_back(std::move(arg));
            }
        }
        define(fn_name, func_ptr);
    }

    // --- lower_func_body: Implementation ---
    void lower_func_body(const Stmt& s) {
        auto& func = s.kind.func.value();
        ir::Function* ir_func = global_functions[func.ident.name()];
        if (!ir_func || func.body.stmts.empty()) return;
        this->fn_ = ir_func;
        this->bb_ = ir_func->add_block("entry");
        this->scopes_.push_back({}); 
        define(func.ident.name(), ir_func);

        // Define parameters in the new scope
        for (size_t i = 0; i < func.params.size(); ++i) {
            if (i < ir_func->params.size()) {
                define(func.params[i].name(), ir_func->params[i].get());
            }
        }

        // Lower actual code
        for (auto& body_stmt : func.body.stmts) {
            lower_stmt(*body_stmt); 
        }

        // Cleanup
        this->scopes_.pop_back();
        this->fn_ = nullptr;
        this->bb_ = nullptr;
    }
 
    // ── Cursor ────────────────────────────────────────────────────────────────
    void set_function(Function* fn) { fn_ = fn; }
    void set_block(BasicBlock* bb)  { bb_ = bb; }
    BasicBlock* current_block() const { return bb_; }
    Function*   current_fn()    const { return fn_; }
 
    // ── Scope ─────────────────────────────────────────────────────────────────
    void push_scope() { scopes_.push_back({}); }
    void pop_scope()  { if (!scopes_.empty()) scopes_.pop_back(); }
 
    void define(const std::string& name, Value* v) {
        if (scopes_.empty()) push_scope();
        scopes_.back()[name] = v;
    }
    Value* lookup(const std::string& name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(name); if (f != it->end()) return f->second;
        }
        return nullptr;
    }
 
    // ── SSA names ─────────────────────────────────────────────────────────────
    std::string fresh() { return "%" + std::to_string(counter_++); }
    std::string fresh(const std::string& hint) {
        auto it = name_counts_.find(hint);
        if (it == name_counts_.end()) { name_counts_[hint] = 1; return "%" + hint; }
        return "%" + hint + std::to_string(it->second++);
    }
 
    // ── Emit ──────────────────────────────────────────────────────────────────
    template<typename T, typename... Args>
    T* emit(Args&&... args) {
        if (!bb_) throw std::logic_error("IRBuilder: no active block");
        T* inst = bb_->emit<T>(std::forward<Args>(args)...);
        if (inst && inst->type) {
            bool is_tensor_infer = (inst->type->kind == Type::Kind::Tensor && 
                                    inst->type->elem_type()->is_infer());
            if (inst->type->is_infer() || is_tensor_infer) {
                std::cerr << "[DEBUG] Unresolved type in IR emission!\n"
                        << "  Instruction: " << inst->name << "\n"
                        << "  Detected Type: " << inst->type->str() << "\n";
                // Helpful if you have a way to dump the current function/block
                // std::cerr << "  Context: " << bb_->name << std::endl;
            }
        }
        
        return inst;
    }
 
    // ── Public lowering ───────────────────────────────────────────────────────
 
    Value* lower_expr(const Expr& e)
    {
        switch (e.kind.tag) {
        case ExprKind::Tag::Lit:        return lower_lit(e);
        case ExprKind::Tag::Id:         return lower_id(e);
        case ExprKind::Tag::Binary:     return lower_binary(e);
        case ExprKind::Tag::Unary:      return lower_unary(e);
        case ExprKind::Tag::Assign:     return lower_assign(e);
        case ExprKind::Tag::Call:       return lower_call(e);
        case ExprKind::Tag::Index:      return lower_index(e);
        case ExprKind::Tag::Field:      return lower_field_expr(e);
        case ExprKind::Tag::Scope:      return lower_scope_expr(e);
        case ExprKind::Tag::Spawn:      return lower_spawn(e);
        case ExprKind::Tag::Await:      return lower_await(e);
        case ExprKind::Tag::Grad:       return lower_grad(e);
        case ExprKind::Tag::If:         return lower_if_expr(e);
        case ExprKind::Tag::Block:      return lower_block_expr(e);
        case ExprKind::Tag::Pipe:       return lower_pipe(e);
        case ExprKind::Tag::Range:      return lower_range(e);
        case ExprKind::Tag::ChannelSend:return lower_channel_send(e);
        case ExprKind::Tag::ArrayLit:   return lower_elements_lit(e);
        case ExprKind::Tag::TensorLit:  return lower_tensor_lit(e);
        case ExprKind::Tag::TupleLit:
        case ExprKind::Tag::SetLit:
        case ExprKind::Tag::QueueLit:
        case ExprKind::Tag::StackLit:   return lower_elements_lit(e);
        case ExprKind::Tag::MapLit:     return lower_map_lit(e);
        case ExprKind::Tag::FnExpr:     return lower_fn_expr(e);
        case ExprKind::Tag::Match:      return lower_match_expr(e);
        case ExprKind::Tag::StructLit:  return lower_struct_lit(e);
        }
        throw std::runtime_error("IRBuilder: unhandled ExprKind::Tag");
    }
 
    Value* lower_stmt(const Stmt& s)
    {
        switch (s.kind.tag) {
        case StmtKind::Tag::Let:      lower_let(s);       return nullptr;
        case StmtKind::Tag::Func:     lower_func_stmt(s); return nullptr;
        case StmtKind::Tag::Return:   lower_return(s);    return nullptr;
        case StmtKind::Tag::If:       lower_if_stmt(s);   return nullptr;
        case StmtKind::Tag::While:    lower_while(s);     return nullptr;
        case StmtKind::Tag::For:      lower_for(s);       return nullptr;
        case StmtKind::Tag::Compound: lower_compound(s.kind.compound); return nullptr;
        case StmtKind::Tag::Expr:
            if (s.kind.expr) return lower_expr(*s.kind.expr);
            return nullptr;
        case StmtKind::Tag::Break:    lower_break();      return nullptr;
        case StmtKind::Tag::Continue: lower_continue();   return nullptr;
        case StmtKind::Tag::Match:    lower_match_stmt(s);return nullptr;
        case StmtKind::Tag::Spawn:
            if (s.kind.spawn_fn) lower_stmt(*s.kind.spawn_fn);
            return nullptr;
        case StmtKind::Tag::Else: {
            lower_compound(s.kind.else_body);
            return nullptr;
        }
        case StmtKind::Tag::Import:   // handled by ImportResolver
        case StmtKind::Tag::Struct:   // type declaration, no IR
        default:
            return nullptr;
        }
    }
 
private:
    ir::IRModule* mod_;
    Function*   fn_      = nullptr;
    BasicBlock* bb_      = nullptr;
    int         counter_ = 0;
    std::unordered_map<std::string, int> name_counts_;
    std::vector<Scope> scopes_;
    std::vector<BasicBlock*> loop_exit_stack_;
    std::vector<BasicBlock*> loop_header_stack_;
    std::unordered_map<Value*, ValuePtr> ptr_cache_;
    std::unordered_map<std::string, Function*> global_functions;
 
    // Non-owning shared_ptr alias so raw Value* can be passed as ValuePtr
    ValuePtr vp(Value* v) {
        if (!v) return nullptr;
        auto it = ptr_cache_.find(v);
        if (it != ptr_cache_.end()) return it->second;
        auto sp = std::shared_ptr<Value>(v, [](Value*){});
        return ptr_cache_[v] = sp;
    }
    Value* keep(std::shared_ptr<Value> c) {
        Value* r = c.get(); ptr_cache_[r] = std::move(c); return r;
    }
 
    // ── TyKind → TypePtr ──────────────────────────────────────────────────────
    static TypePtr ty(TyKind tk) {
        switch (tk) {
        case TyKind::Void:   return Type::void_();
        case TyKind::Bool:   return Type::bool_();
        case TyKind::I32:    return Type::i32();
        case TyKind::I64:    return Type::i64();
        case TyKind::F32:    return Type::f32();
        case TyKind::F64:    return Type::f64();
        case TyKind::Str:    return Type::str_();
        case TyKind::Tensor: return Type::tensor(Type::f32());
        case TyKind::Array:  return Type::array(Type::infer());
        default:             return Type::infer();
        }
    }
 
    TypePtr expr_type(const Expr& e) const {
        if (e.resolved_type) return e.resolved_type;
        if (e.kind.tag == ExprKind::Tag::Id) return ty(e.kind.id.ty_kind());
        return Type::infer();
    }
 
    // ── Top-level ─────────────────────────────────────────────────────────────
    void lower_top_level(const Stmt& s) {
        switch (s.kind.tag) {
        case StmtKind::Tag::Func:   lower_func_stmt(s); break;
        case StmtKind::Tag::Let:    lower_global_let(s); break;
        default: break;
        }
    }
 
    void lower_func_stmt(const Stmt& s) {
        if (s.kind.func) lower_func(*s.kind.func);
    }
 
    void lower_func(const Func& f) {
        std::cout << "Function " << f.ident.name() << " ty_kind: " 
              << (int)f.ident.ty_kind() << std::endl;
        std::vector<TypePtr> ptypes;
        for (auto& p : f.params) ptypes.push_back(ty(p.ty_kind()));
        TypePtr ret_ty  = ty(f.ident.ty_kind());
        TypePtr fn_type = Type::fn(ptypes, ret_ty);

        const std::string& raw = f.ident.name();
        std::string fn_name = (raw.empty() || raw[0] != '@') ? "@" + raw : raw;
        Function* fn = mod_->add_function(fn_name, fn_type, f.is_async);
        global_functions[f.ident.name()] = fn;
        set_function(fn); push_scope();
        set_block(fn->create_entry());
 
        for (auto& p : f.params) {
            Argument* a = fn->add_param("%" + p.name(), ty(p.ty_kind()));
            define(p.name(), a);
        }
        lower_compound(f.body);
        if (!bb_->is_terminated()) emit<ReturnInst>();
        pop_scope(); set_function(nullptr); set_block(nullptr);
    }
 
    void lower_global_let(const Stmt& s) {
        if (!s.kind.let_expr) return;
        auto cv = make_const_from_expr(*s.kind.let_expr);
        if (!cv) return;
        const std::string& raw = s.kind.let_ident.name();
        cv->name = (raw.empty() || raw[0] != '@') ? "@" + raw : raw;
        ir::Value* global_ptr = cv.get();
        mod_->add_global(std::move(cv));
        define(s.kind.let_ident.name(), global_ptr);
    }
 
    void lower_compound(const Compound& c) {
        push_scope();
        for (auto& stmt : c.stmts) {
            if (stmt) lower_stmt(*stmt);
        }
        if (c.tail_expr) lower_expr(*c.tail_expr);
        pop_scope();
    }
 
    // ── Statements ────────────────────────────────────────────────────────────
 
    void lower_let(const Stmt& s) {
        const Ident& id = s.kind.let_ident;
        if (!s.kind.let_expr) {
            // Uninitialized mutable — emit alloca
            auto* alloca = emit<AllocaInst>(fresh(id.name()), ty(id.ty_kind()));
            define(id.name(), alloca);
            return;
        }
        Value* rhs = lower_expr(*s.kind.let_expr);
        rhs->name = "%" + id.name();
        define(id.name(), rhs);
    }
 
    void lower_return(const Stmt& s) {
        if (s.kind.ret_expr) emit<ReturnInst>(vp(lower_expr(*s.kind.ret_expr)));
        else                  emit<ReturnInst>();
    }
 
    void lower_if_stmt(const Stmt& s) {
        Value* cond     = lower_expr(*s.kind.if_cond);
        auto* then_bb   = fn_->add_block(fresh_label("if.true"));
        auto* merge_bb  = fn_->add_block(fresh_label("if.merge"));
        bool has_else = s.kind.else_or_else_if || !s.kind.else_body.stmts.empty();
        auto* else_bb   = has_else
                        ? fn_->add_block(fresh_label("if.false"))
                        : nullptr;
 
        emit<CondBranchInst>(vp(cond), then_bb, else_bb ? else_bb : merge_bb);
 
        set_block(then_bb);
        lower_compound(s.kind.if_body);
        if (!bb_->is_terminated()) emit<BranchInst>(merge_bb);
 
        if (else_bb) {
            set_block(else_bb);
            if (s.kind.else_or_else_if) {
                lower_stmt(*s.kind.else_or_else_if);
            } else {
                lower_compound(s.kind.else_body);
            }
            if (!bb_->is_terminated()) emit<BranchInst>(merge_bb);
        }
        set_block(merge_bb);
    }
 
    void lower_while(const Stmt& s) {
        auto* header = fn_->add_block(fresh_label("while.cond"));
        auto* body   = fn_->add_block(fresh_label("while.body"));
        auto* exit   = fn_->add_block(fresh_label("while.exit"));
        emit<BranchInst>(header);
 
        set_block(header);
        if (s.kind.while_cond) {
            emit<CondBranchInst>(vp(lower_expr(*s.kind.while_cond)), body, exit);
        } else {
            emit<BranchInst>(body); // infinite loop
        }
 
        loop_exit_stack_.push_back(exit);
        loop_header_stack_.push_back(header);
        set_block(body);
        lower_compound(s.kind.while_body);
        if (!bb_->is_terminated()) emit<BranchInst>(header);
        loop_exit_stack_.pop_back();
        loop_header_stack_.pop_back();
        set_block(exit);
    }
 
    void lower_for(const Stmt& s) {
        Value* iter_val = lower_expr(*s.kind.for_iter);
        auto* header = fn_->add_block(fresh_label("for.cond"));
        auto* body   = fn_->add_block(fresh_label("for.body"));
        auto* exit   = fn_->add_block(fresh_label("for.exit"));
        emit<BranchInst>(header);
 
        // Placeholder has_next condition
        set_block(header);
        auto placeholder = std::make_shared<ConstantBool>(true);
        keep(placeholder);
        emit<CondBranchInst>(vp(placeholder.get()), body, exit);
 
        loop_exit_stack_.push_back(exit);
        loop_header_stack_.push_back(header);
        set_block(body); push_scope();
        // Bind loop variable as placeholder
        auto lv = std::make_shared<Value>("%" + s.kind.for_var, Type::infer());
        keep(lv); define(s.kind.for_var, lv.get());
        lower_compound(s.kind.for_body);
        pop_scope();
        if (!bb_->is_terminated()) emit<BranchInst>(header);
        loop_exit_stack_.pop_back();
        loop_header_stack_.pop_back();
        set_block(exit);
        (void)iter_val;
    }
 
    void lower_break() {
        if (loop_exit_stack_.empty())
            throw std::runtime_error("IRBuilder: break outside loop");
        emit<BranchInst>(loop_exit_stack_.back());
        set_block(fn_->add_block(fresh_label("dead")));
    }
 
    void lower_continue() {
        if (loop_header_stack_.empty())
            throw std::runtime_error("IRBuilder: continue outside loop");
        emit<BranchInst>(loop_header_stack_.back());
        set_block(fn_->add_block(fresh_label("dead")));
    }
 
    void lower_match_stmt(const Stmt& s) {
        lower_expr(*s.kind.match_subject);
        // Full pattern compilation is a separate pass
    }
 
    // ── Expressions ───────────────────────────────────────────────────────────
 
    Value* lower_lit(const Expr& e) {
        const LitKind& lit = e.kind.lit;
        switch (lit.tag) {
        case LitKind::Tag::Int:
            return keep(std::make_shared<ConstantInt>(
                std::stoll(lit.str_val),
                e.resolved_type ? e.resolved_type : Type::i64()));
        case LitKind::Tag::Float:
            return keep(std::make_shared<ConstantFloat>(
                std::stod(lit.str_val),
                e.resolved_type ? e.resolved_type : Type::f64()));
        case LitKind::Tag::Bool:
            return keep(std::make_shared<ConstantBool>(lit.bool_val));
        case LitKind::Tag::Str:
            return keep(std::make_shared<ConstantString>(lit.str_val));
        }
        throw std::runtime_error("IRBuilder: unknown LitKind");
    }
 
    Value* lower_id(const Expr& e) {
        const std::string& name = e.kind.id.name();
        Value* v = lookup(name);
        if (!v && mod_) {
            std::string fn_name = (name.empty() || name[0] != '@') ? "@" + name : name;
            v = mod_->find_function(fn_name);
        }
        if (!v) throw std::runtime_error("IRBuilder: undefined name '" + name + "'");
        if (dynamic_cast<AllocaInst*>(v))
            return emit<LoadInst>(fresh(name), v->type, vp(v));
        return v;
    }
 
    Value* lower_binary(const Expr& e) {
        if (!e.kind.lhs || !e.kind.rhs) {
            throw std::runtime_error("IRBuilder: Malformed binary expression (null operand)");
        }
        TypePtr ty_  = expr_type(e);
        // Comparison → CmpInst
        switch (e.kind.bin_op) {
        case BinOp::Eq:     { auto l=lower_expr(*e.kind.lhs),r=lower_expr(*e.kind.rhs); return emit<CmpInst>(fresh(),CmpCode::Eq,vp(l),vp(r)); }
        case BinOp::Neq:    { auto l=lower_expr(*e.kind.lhs),r=lower_expr(*e.kind.rhs); return emit<CmpInst>(fresh(),CmpCode::Ne,vp(l),vp(r)); }
        case BinOp::Lt:     { auto l=lower_expr(*e.kind.lhs),r=lower_expr(*e.kind.rhs); return emit<CmpInst>(fresh(),CmpCode::Lt,vp(l),vp(r)); }
        case BinOp::Lte:    { auto l=lower_expr(*e.kind.lhs),r=lower_expr(*e.kind.rhs); return emit<CmpInst>(fresh(),CmpCode::Le,vp(l),vp(r)); }
        case BinOp::Gt:     { auto l=lower_expr(*e.kind.lhs),r=lower_expr(*e.kind.rhs); return emit<CmpInst>(fresh(),CmpCode::Gt,vp(l),vp(r)); }
        case BinOp::Gte:    { auto l=lower_expr(*e.kind.lhs),r=lower_expr(*e.kind.rhs); return emit<CmpInst>(fresh(),CmpCode::Ge,vp(l),vp(r)); }
        case BinOp::MatMul: {
            auto l=lower_expr(*e.kind.lhs),r=lower_expr(*e.kind.rhs);
            return emit<TensorOpInst>(fresh(),ty_,TensorOpCode::MatMul,std::vector<ValuePtr>{vp(l),vp(r)});
        }
        default: break;
        }
        // Arithmetic / logical → BinOpInst
        Value* lhs = lower_expr(*e.kind.lhs);
        Value* rhs = lower_expr(*e.kind.rhs);
        bool f = ty_ && ty_->is_float();
        BinOpCode op;
        switch (e.kind.bin_op) {
        case BinOp::Add: op = f ? BinOpCode::FAdd : BinOpCode::Add; break;
        case BinOp::Sub: op = f ? BinOpCode::FSub : BinOpCode::Sub; break;
        case BinOp::Mul: op = f ? BinOpCode::FMul : BinOpCode::Mul; break;
        case BinOp::Div: op = f ? BinOpCode::FDiv : BinOpCode::Div; break;
        case BinOp::And: op = BinOpCode::And; break;
        case BinOp::Or:  op = BinOpCode::Or;  break;
        default: throw std::runtime_error("IRBuilder: unhandled BinOp");
        }
        TypePtr res_ty = lhs->type; 
        if (res_ty->kind == Type::Kind::Tensor) {
            auto elem = res_ty->elem_type();
            if (!elem || elem->is_infer()) res_ty = rhs->type; // Fallback to right side
        }
        return emit<BinOpInst>(fresh(), res_ty, op, vp(lhs), vp(rhs));
    }
 
    Value* lower_unary(const Expr& e) {
        Value* operand = lower_expr(*e.kind.operand);
        TypePtr ty_ = expr_type(e);
        bool f = ty_ && ty_->is_float();
        UnOpCode op = (e.kind.unary_op == UnaryOp::Neg)
                    ? (f ? UnOpCode::FNeg : UnOpCode::Neg)
                    : UnOpCode::Not;
        return emit<UnOpInst>(fresh(), ty_, op, vp(operand));
    }
 
    Value* lower_assign(const Expr& e) {
        if (!e.kind.lhs || !e.kind.rhs) return nullptr;
        Value* rhs_val = nullptr;
        if (e.kind.bin_op == BinOp::Assign) {
            rhs_val = lower_expr(*e.kind.rhs);
        } else {
            Value* cur = lower_expr(*e.kind.lhs);
            Value* rhs = lower_expr(*e.kind.rhs);
            TypePtr ty_ = expr_type(e);
            bool f = ty_ && ty_->is_float();
            BinOpCode op;
            switch (e.kind.bin_op) {
            case BinOp::AddAssign: op = f ? BinOpCode::FAdd : BinOpCode::Add; break;
            case BinOp::SubAssign: op = f ? BinOpCode::FSub : BinOpCode::Sub; break;
            case BinOp::MulAssign: op = f ? BinOpCode::FMul : BinOpCode::Mul; break;
            case BinOp::DivAssign: op = f ? BinOpCode::FDiv : BinOpCode::Div; break;
            default: throw std::runtime_error("IRBuilder: unknown compound assign");
            }
            rhs_val = emit<BinOpInst>(fresh(), ty_, op, vp(cur), vp(rhs));
        }
        store_to_lvalue(*e.kind.lhs, rhs_val);
        return rhs_val;
    }
 
    void store_to_lvalue(const Expr& lval, Value* val) {
        if (lval.kind.tag == ExprKind::Tag::Id) {
            const std::string& name = lval.kind.id.name();
            Value* slot = lookup(name);
            if (!slot) throw std::runtime_error("IRBuilder: assign to undefined '" + name + "'");
            if (dynamic_cast<AllocaInst*>(slot)) {
                emit<StoreInst>(vp(val), vp(slot));
            } else {
                // Promote to mutable slot
                auto* alloca = emit<AllocaInst>(fresh(name + ".slot"), val->type);
                emit<StoreInst>(vp(val), vp(alloca));
                define(name, alloca);
            }
        } else if (lval.kind.tag == ExprKind::Tag::Index) {
            Value* base  = lower_expr(*lval.kind.target);
            Value* index = lower_expr(*lval.kind.index);
            emit<TensorOpInst>("", Type::void_(), TensorOpCode::Scatter,
                std::vector<ValuePtr>{vp(base), vp(index), vp(val)});
        } else {
            throw std::runtime_error("IRBuilder: unsupported lvalue");
        }
    }
 
    Value* lower_call(const Expr& e) {
        const Expr& callee = *e.kind.callee;
        TypePtr ret = expr_type(e);
        std::vector<ValuePtr> args;
        if (callee.kind.tag == ExprKind::Tag::Scope) {
            std::string ns = callee.kind.target->kind.id.name();
            if (ns == "ts") {
                return lower_tensor_call(callee, e.kind.args, ret);
            } 
            if (ns == "std") {
                return lower_std_call(callee.kind.member, e.kind.args, ret);
            }
        }
        for (auto& a : e.kind.args) {
            args.push_back(vp(lower_expr(*a)));
        }
        if (callee.kind.tag == ExprKind::Tag::Id) {
            std::string name = callee.kind.id.name();
            auto it = global_functions.find(name);
            if (it != global_functions.end()) {
                Function* fn = it->second;
                return emit<CallInst>(ret->is_void() ? "" : fresh(), ret, vp(fn), std::move(args));
            }
        }
        Value* callee_val = lower_expr(callee); 
        return emit<CallInst>(ret->is_void() ? "" : fresh(), ret, vp(callee_val), std::move(args));
    }
 
    Value* lower_tensor_call(const Expr& scope_callee,
                              const std::vector<ExprPtr>& ast_args,
                              TypePtr ret_type)
    {
        TensorOpCode op = resolve_tensor_op(scope_callee.kind.member);
        std::vector<ValuePtr> args;
        for (auto& a : ast_args) args.push_back(vp(lower_expr(*a)));
        TypePtr effective_ty = ret_type;
        if (effective_ty->is_infer() || effective_ty->elem_type()->is_infer()) {
            // Default: try to copy element type from first good tensor arg
            TypePtr elem_ty = Type::f32();  // safe fallback

            for (const auto& arg : args) {
                if (arg->type && arg->type->kind == Type::Kind::Tensor) {
                    TypePtr el = arg->type->elem_type();
                    if (el && !el->is_infer()) {
                        elem_ty = el;
                        break;
                    }
                }
            }

            // Most tensor ops preserve tensor + element type
            effective_ty = Type::tensor(elem_ty);

            // Special case: matmul (still tensor)
            if (op == TensorOpCode::MatMul) {
                // could add shape inference later, for now just ensure tensor<f32>
                effective_ty = Type::tensor(elem_ty);
            }
        }
        // For ops known to return scalar (reduce-to-number)
        if (op == TensorOpCode::Sum || op == TensorOpCode::Mean || op == TensorOpCode::Max ||
            op == TensorOpCode::Min || op == TensorOpCode::Prod) {
            effective_ty = Type::f32();
        }
        return emit<TensorOpInst>(op_is_void(op) ? "" : fresh(), effective_ty, op, std::move(args));
    }

    Value* lower_std_call(const std::string& func_name, const std::vector<ExprPtr>& args, TypePtr ret) {
        std::vector<ValuePtr> ir_args;
        for (auto& a : args) ir_args.push_back(vp(lower_expr(*a)));
        // Look for a built-in function named "@std.println" or similar
        Function* std_fn = nullptr;
        std::string mangled_name = "@std." + func_name;
        if (global_functions.count(mangled_name)) {
            std_fn = global_functions[mangled_name];
        } else {
            std::vector<TypePtr> arg_types;
            for (auto& v : ir_args) arg_types.push_back(v->type);
            std_fn = mod_->add_function(mangled_name, Type::fn(arg_types, ret), false);
            global_functions[mangled_name] = std_fn;
        }

        return emit<CallInst>(ret->is_void() ? "" : fresh(), ret, vp(std_fn), std::move(ir_args), false);
    }
 
    Value* lower_index(const Expr& e) {
        Value* base  = lower_expr(*e.kind.target);
        Value* index = lower_expr(*e.kind.index);
        return emit<TensorOpInst>(fresh(), expr_type(e), TensorOpCode::Select,
            std::vector<ValuePtr>{vp(base), vp(index)});
    }
 
    Value* lower_field_expr(const Expr& e) {
        Value* obj = lower_expr(*e.kind.target);
        std::string qname = obj->name + "." + e.kind.member;
        if (Value* v = lookup(qname)) return v;
        return emit<LoadInst>(fresh(e.kind.member), expr_type(e), vp(obj));
    }
 
    Value* lower_scope_expr(const Expr& e) {
        // e.kind.target must be an Id (the namespace)
        const std::string& ns   = e.kind.target->kind.id.name();
        const std::string& item = e.kind.member;
        if (Value* v = lookup(ns + "::" + item)) return v;
        throw std::runtime_error("IRBuilder: unresolved '" + ns + "::" + item + "'");
    }
 
    Value* lower_spawn(const Expr& e) {
        Value* task = lower_expr(*e.kind.spawned_expr);
        return emit<SpawnInst>(fresh("h"), task->type, vp(task));
    }
 
    Value* lower_await(const Expr& e) {
        if (!e.kind.operand) {
            throw std::runtime_error("IRBuilder Error: 'await' expression is missing its target operand.");
        }
        Value* handle = lower_expr(*e.kind.operand);
        TypePtr awaited_ty = handle->type;
        return emit<AwaitInst>(fresh("await_tmp"), awaited_ty, vp(handle));
    }
 
    Value* lower_grad(const Expr& e) {
        Value* loss   = lower_expr(*e.kind.grad_loss);
        Value* params = lower_expr(*e.kind.grad_params);
        return emit<TensorOpInst>(fresh(), expr_type(e), TensorOpCode::Grad,
            std::vector<ValuePtr>{vp(loss), vp(params)});
    }
 
    Value* lower_if_expr(const Expr& e) {
        Value* cond = lower_expr(*e.kind.condition);
        auto* then_bb  = fn_->add_block(fresh_label("ifexpr.true"));
        auto* else_bb  = fn_->add_block(fresh_label("ifexpr.false"));
        auto* merge_bb = fn_->add_block(fresh_label("ifexpr.merge"));
        emit<CondBranchInst>(vp(cond), then_bb, else_bb);
 
        set_block(then_bb);
        Value* then_val = e.kind.then_branch ? lower_expr(*e.kind.then_branch) : nullptr;
        BasicBlock* then_exit = bb_;
        if (!bb_->is_terminated()) emit<BranchInst>(merge_bb);
 
        set_block(else_bb);
        Value* else_val = e.kind.else_branch ? lower_expr(*e.kind.else_branch) : nullptr;
        BasicBlock* else_exit = bb_;
        if (!bb_->is_terminated()) emit<BranchInst>(merge_bb);
 
        set_block(merge_bb);
        if (then_val && else_val) {
            auto* phi = emit<PhiInst>(fresh("ifval"), expr_type(e));
            phi->add_incoming(vp(then_val), then_exit);
            phi->add_incoming(vp(else_val), else_exit);
            return phi;
        }
        return then_val ? then_val : else_val;
    }
 
    Value* lower_block_expr(const Expr& e) {
        push_scope();
        for (auto& stmt : e.kind.block.stmts) lower_stmt(*stmt);
        Value* result = nullptr;
        if (e.kind.block.tail_expr) result = lower_expr(*e.kind.block.tail_expr);
        pop_scope();
        return result;
    }
 
    Value* lower_pipe(const Expr& e) {
        Value* lhs_val = lower_expr(*e.kind.pipe_lhs);
        if (e.kind.pipe_rhs->kind.tag == ExprKind::Tag::Call) {
            const Expr& call_e = *e.kind.pipe_rhs;
            Value* callee_val = lower_expr(*call_e.kind.callee);
            std::vector<ValuePtr> args;
            args.push_back(vp(lhs_val));
            for (auto& a : call_e.kind.args) args.push_back(vp(lower_expr(*a)));
            TypePtr ret = expr_type(e);
            return emit<CallInst>(ret->is_void() ? "" : fresh(), ret, vp(callee_val), std::move(args));
        } else {
            Value* fn_val = lower_expr(*e.kind.pipe_rhs);
            TypePtr ret = expr_type(e);
            return emit<CallInst>(ret->is_void() ? "" : fresh(), ret,
                vp(fn_val), std::vector<ValuePtr>{vp(lhs_val)});
        }
    }
 
    Value* lower_range(const Expr& e) {
        Value* lo = lower_expr(*e.kind.lhs);
        Value* hi = lower_expr(*e.kind.rhs);
        return emit<TensorOpInst>(fresh(), expr_type(e), TensorOpCode::Arange,
            std::vector<ValuePtr>{vp(lo), vp(hi)});
    }
 
    Value* lower_channel_send(const Expr& e) {
        Value* ch  = lower_expr(*e.kind.channel);
        Value* val = lower_expr(*e.kind.send_val);
        auto callee = std::make_shared<Value>("__channel_send", Type::fn({}, Type::void_()));
        keep(callee);
        emit<CallInst>("", Type::void_(), vp(callee.get()),
            std::vector<ValuePtr>{vp(ch), vp(val)});
        return nullptr;
    }
 
    Value* lower_elements_lit(const Expr& e) {
        std::vector<ValuePtr> elems;
        for (auto& el : e.kind.elements) elems.push_back(vp(lower_expr(*el)));
        return emit<TensorOpInst>(fresh(), expr_type(e), TensorOpCode::FromList, std::move(elems));
    }
 
    Value* lower_tensor_lit(const Expr& e) {
        std::vector<ValuePtr> elems;
        for (auto& row : e.kind.rows)
            for (auto& el : row) elems.push_back(vp(lower_expr(*el)));
        return emit<TensorOpInst>(fresh(), expr_type(e), TensorOpCode::FromList, std::move(elems));
    }
 
    Value* lower_map_lit(const Expr& e) {
        std::vector<ValuePtr> elems;
        for (auto& [k, v] : e.kind.map_pairs) {
            elems.push_back(vp(lower_expr(*k)));
            elems.push_back(vp(lower_expr(*v)));
        }
        return emit<TensorOpInst>(fresh(), expr_type(e), TensorOpCode::FromList, std::move(elems));
    }
 
    Value* lower_fn_expr(const Expr& e) {
        std::string name = fresh_label("__lambda");
        std::vector<TypePtr> ptypes;
        for (auto& [pname, pk] : e.kind.fn_params) ptypes.push_back(ty(pk));
        TypePtr ret_ty  = ty(e.kind.fn_ret_type);
        TypePtr fn_type = Type::fn(ptypes, ret_ty);
        Function* fn = mod_->add_function("@" + name, fn_type, e.kind.is_async_fn);
 
        // Stash builder state
        Function* sfn = fn_; BasicBlock* sbb = bb_;
        auto sscopes = scopes_; scopes_.clear(); int sc = counter_;
 
        set_function(fn); push_scope();
        set_block(fn->create_entry());
        for (auto& [pname, pk] : e.kind.fn_params) {
            Argument* a = fn->add_param("%" + pname, ty(pk));
            define(pname, a);
        }
        lower_compound(e.kind.fn_body);
        if (!bb_->is_terminated()) emit<ReturnInst>();
        pop_scope();
 
        fn_ = sfn; bb_ = sbb; scopes_ = std::move(sscopes); counter_ = sc;
 
        auto fn_val = std::make_shared<Value>("@" + name, fn_type);
        keep(fn_val);
        return fn_val.get();
    }
 
    Value* lower_match_expr(const Expr& e) {
        lower_expr(*e.kind.match_subject);
        auto undef = std::make_shared<Value>(fresh("match_undef"), expr_type(e));
        keep(undef); return undef.get();
    }
 
    Value* lower_struct_lit(const Expr& e) {
        TypePtr ty_ = expr_type(e);
        auto* alloca = emit<AllocaInst>(fresh(e.kind.struct_init_name), ty_);
        for (auto& [fname, fexpr] : e.kind.struct_init_fields) {
            Value* fval = lower_expr(*fexpr);
            std::string qname = alloca->name + "." + fname;
            auto* fslot = emit<AllocaInst>(qname, fval->type);
            emit<StoreInst>(vp(fval), vp(fslot));
            define(qname, fslot);
        }
        return alloca;
    }
 
    // ── Helpers ───────────────────────────────────────────────────────────────
 
    std::string fresh_label(const std::string& base) {
        auto it = name_counts_.find(base);
        if (it == name_counts_.end()) { name_counts_[base] = 1; return base; }
        return base + std::to_string(it->second++);
    }
 
    std::shared_ptr<Value> make_const_from_expr(const Expr& e) {
        if (e.kind.tag != ExprKind::Tag::Lit) return nullptr;
        const LitKind& lit = e.kind.lit;
        switch (lit.tag) {
        case LitKind::Tag::Int:
            return std::make_shared<ConstantInt>(std::stoll(lit.str_val),
                e.resolved_type ? e.resolved_type : Type::i64());
        case LitKind::Tag::Float:
            return std::make_shared<ConstantFloat>(std::stod(lit.str_val),
                e.resolved_type ? e.resolved_type : Type::f64());
        case LitKind::Tag::Bool:
            return std::make_shared<ConstantBool>(lit.bool_val);
        case LitKind::Tag::Str:
            return std::make_shared<ConstantString>(lit.str_val);
        }
        return nullptr;
    }
 
    static TensorOpCode resolve_tensor_op(const std::string& n) {
        static const std::unordered_map<std::string,TensorOpCode> T = {
            {"zeros",TensorOpCode::Zeros},{"ones",TensorOpCode::Ones},
            {"full",TensorOpCode::Full},{"eye",TensorOpCode::Eye},
            {"arange",TensorOpCode::Arange},{"linspace",TensorOpCode::Linspace},
            {"rand",TensorOpCode::Rand},{"randn",TensorOpCode::Randn},
            {"randint",TensorOpCode::RandInt},{"from_list",TensorOpCode::FromList},
            {"reshape",TensorOpCode::Reshape},{"view",TensorOpCode::View},
            {"flatten",TensorOpCode::Flatten},{"squeeze",TensorOpCode::Squeeze},
            {"unsqueeze",TensorOpCode::Unsqueeze},{"transpose",TensorOpCode::Transpose},
            {"permute",TensorOpCode::Permute},{"contiguous",TensorOpCode::Contiguous},
            {"clone",TensorOpCode::Clone},{"slice",TensorOpCode::Slice},
            {"select",TensorOpCode::Select},{"cat",TensorOpCode::Cat},
            {"stack",TensorOpCode::Stack},{"split",TensorOpCode::Split},
            {"chunk",TensorOpCode::Chunk},{"tile",TensorOpCode::Tile},
            {"repeat",TensorOpCode::Repeat},{"pad",TensorOpCode::Pad},
            {"sum",TensorOpCode::Sum},{"mean",TensorOpCode::Mean},
            {"max",TensorOpCode::Max},{"min",TensorOpCode::Min},
            {"prod",TensorOpCode::Prod},{"norm",TensorOpCode::Norm},
            {"std",TensorOpCode::Std},{"var",TensorOpCode::Var},
            {"median",TensorOpCode::Median},{"argmax",TensorOpCode::ArgMax},
            {"argmin",TensorOpCode::ArgMin},
            {"exp",TensorOpCode::Exp},{"log",TensorOpCode::Log},
            {"log2",TensorOpCode::Log2},{"log1p",TensorOpCode::Log1p},
            {"sqrt",TensorOpCode::Sqrt},{"rsqrt",TensorOpCode::Rsqrt},
            {"abs",TensorOpCode::Abs},{"sign",TensorOpCode::Sign},
            {"sin",TensorOpCode::Sin},{"cos",TensorOpCode::Cos},
            {"tan",TensorOpCode::Tan},{"floor",TensorOpCode::Floor},
            {"ceil",TensorOpCode::Ceil},{"round",TensorOpCode::Round},
            {"pow",TensorOpCode::Pow},{"clamp",TensorOpCode::Clamp},
            {"lerp",TensorOpCode::Lerp},{"relu",TensorOpCode::Relu},
            {"relu6",TensorOpCode::Relu6},{"silu",TensorOpCode::Silu},
            {"gelu",TensorOpCode::Gelu},{"sigmoid",TensorOpCode::Sigmoid},
            {"tanh",TensorOpCode::Tanh},{"softmax",TensorOpCode::Softmax},
            {"log_softmax",TensorOpCode::LogSoftmax},
            {"leaky_relu",TensorOpCode::LeakyRelu},{"elu",TensorOpCode::Elu},
            {"selu",TensorOpCode::Selu},{"prelu",TensorOpCode::Prelu},
            {"dot",TensorOpCode::Dot},{"matmul",TensorOpCode::MatMul},
            {"bmm",TensorOpCode::Bmm},{"outer",TensorOpCode::Outer},
            {"cross",TensorOpCode::Cross},{"inverse",TensorOpCode::Inverse},
            {"det",TensorOpCode::Det},{"trace",TensorOpCode::Trace},
            {"svd",TensorOpCode::Svd},{"eig",TensorOpCode::Eig},
            {"qr",TensorOpCode::Qr},{"cholesky",TensorOpCode::Cholesky},
            {"solve",TensorOpCode::Solve},{"sort",TensorOpCode::Sort},
            {"argsort",TensorOpCode::ArgSort},{"topk",TensorOpCode::TopK},
            {"gather",TensorOpCode::Gather},{"scatter",TensorOpCode::Scatter},
            {"where",TensorOpCode::Where},{"nonzero",TensorOpCode::NonZero},
            {"backward",TensorOpCode::Backward},{"grad",TensorOpCode::Grad},
            {"no_grad",TensorOpCode::NoGrad},{"detach",TensorOpCode::Detach},
            {"zero_grad",TensorOpCode::ZeroGrad},
            {"requires_grad",TensorOpCode::RequiresGrad},
        };
        auto it = T.find(n);
        if (it == T.end())
            throw std::runtime_error("IRBuilder: unknown tensor op '" + n + "'");
        return it->second;
    }
 
    static bool op_is_void(TensorOpCode op) {
        return op == TensorOpCode::Backward ||
               op == TensorOpCode::ZeroGrad ||
               op == TensorOpCode::NoGrad;
    }
};
 
} // namespace ir