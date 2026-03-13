#pragma once
 
/// ir/IRBuilder.h  — AST structs fully defined before IRBuilder uses them.
 
#include "IRModule.h"
#include "IRPrinter.h"
 
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
 
// ═══════════════════════════════════════════════════════════════
// AST stub types  (replace with #include "ast/AST.h" later)
// ALL structs defined HERE, before IRBuilder, so dynamic_cast works.
// ═══════════════════════════════════════════════════════════════
 
struct ASTNode { TypePtr resolved_type; virtual ~ASTNode() = default; };
 
struct LitIntExpr   : ASTNode { int64_t     value = 0; };
struct LitFloatExpr : ASTNode { double      value = 0.0; };
struct LitBoolExpr  : ASTNode { bool        value = false; };
struct LitStrExpr   : ASTNode { std::string value; };
struct IdentExpr    : ASTNode { std::string name; };
 
struct BinExpr : ASTNode {
    std::string op;
    std::unique_ptr<ASTNode> lhs, rhs;
};
struct UnExpr : ASTNode {
    std::string op;
    std::unique_ptr<ASTNode> operand;
};
struct FieldExpr : ASTNode {
    std::unique_ptr<ASTNode> object;
    std::string module_alias, field;
};
struct IndexExpr : ASTNode {
    std::unique_ptr<ASTNode> base, index;
};
struct CallExpr : ASTNode {
    std::unique_ptr<ASTNode> callee;
    std::vector<std::unique_ptr<ASTNode>> args;
};
struct SpawnExpr : ASTNode { std::unique_ptr<ASTNode> expr; };
struct AwaitExpr : ASTNode { std::unique_ptr<ASTNode> expr; };
 
struct ExprStmt   : ASTNode { std::unique_ptr<ASTNode> expr; };
struct LetStmt    : ASTNode {
    std::string name;
    bool is_mutable = false;
    std::unique_ptr<ASTNode> init;
};
struct ReturnStmt : ASTNode { std::unique_ptr<ASTNode> value; };
struct IfStmt     : ASTNode {
    std::unique_ptr<ASTNode> cond;
    std::vector<std::unique_ptr<ASTNode>> then_body;
    std::unique_ptr<std::vector<std::unique_ptr<ASTNode>>> else_body;
};
struct WhileStmt  : ASTNode {
    std::unique_ptr<ASTNode> cond;
    std::vector<std::unique_ptr<ASTNode>> body;
};
struct BlockStmt  : ASTNode {
    std::vector<std::unique_ptr<ASTNode>> stmts;
};
 
struct ParamDecl { std::string name; TypePtr type; };
struct FnDecl    : ASTNode {
    std::string name;
    bool is_exported = false, is_async = false;
    std::vector<ParamDecl> params;
    std::vector<std::unique_ptr<ASTNode>> body;
};
 
struct Program { std::vector<std::unique_ptr<ASTNode>> stmts; };
 
// ═══════════════════════════════════════════════════════════════
// IRBuilder
// ═══════════════════════════════════════════════════════════════
 
namespace ir {
 
using Scope = std::unordered_map<std::string, Value*>;
 
class IRBuilder {
public:
    IRBuilder() = default;
 
    // ── Main entry point ─────────────────────────────────────────
    std::unique_ptr<IRModule> build(const Program& prog,
                                    const std::string& path = "<stdin>") {
        mod_ = std::make_unique<IRModule>(path);
        for (auto& s : prog.stmts) lower_top_level(*s);
        return std::move(mod_);
    }
 
    // ── Cursor ───────────────────────────────────────────────────
    void set_function(Function* fn) { fn_ = fn; }
    void set_block(BasicBlock* bb)  { bb_ = bb; }
    BasicBlock* current_block() const { return bb_; }
    Function*   current_fn()    const { return fn_; }
 
    // ── Scope ────────────────────────────────────────────────────
    void push_scope() { scopes_.push_back({}); }
    void pop_scope()  { if (!scopes_.empty()) scopes_.pop_back(); }
 
    void define(const std::string& name, Value* v) {
        if (scopes_.empty()) push_scope();
        scopes_.back()[name] = v;
    }
    Value* lookup(const std::string& name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return f->second;
        }
        return nullptr;
    }
 
    // ── SSA names ────────────────────────────────────────────────
    std::string fresh() { return "%" + std::to_string(counter_++); }
    std::string fresh(const std::string& hint) {
        auto it = name_counts_.find(hint);
        if (it == name_counts_.end()) { name_counts_[hint] = 1; return "%" + hint; }
        return "%" + hint + std::to_string(it->second++);
    }
 
    // ── Emit ─────────────────────────────────────────────────────
    template<typename T, typename... Args>
    T* emit(Args&&... args) {
        if (!bb_) throw std::logic_error("IRBuilder: no active block");
        return bb_->emit<T>(std::forward<Args>(args)...);
    }
 
    // ── Expression lowering (public for tests) ───────────────────
    Value* lower_expr(const ASTNode& node) {
        if (auto* e = dynamic_cast<const LitIntExpr*>(&node))   return lower_lit_int(*e);
        if (auto* e = dynamic_cast<const LitFloatExpr*>(&node)) return lower_lit_float(*e);
        if (auto* e = dynamic_cast<const LitBoolExpr*>(&node))  return lower_lit_bool(*e);
        if (auto* e = dynamic_cast<const LitStrExpr*>(&node))   return lower_lit_str(*e);
        if (auto* e = dynamic_cast<const IdentExpr*>(&node))    return lower_ident(*e);
        if (auto* e = dynamic_cast<const BinExpr*>(&node))      return lower_bin(*e);
        if (auto* e = dynamic_cast<const UnExpr*>(&node))       return lower_un(*e);
        if (auto* e = dynamic_cast<const CallExpr*>(&node))     return lower_call(*e);
        if (auto* e = dynamic_cast<const SpawnExpr*>(&node))    return lower_spawn(*e);
        if (auto* e = dynamic_cast<const AwaitExpr*>(&node))    return lower_await(*e);
        if (auto* e = dynamic_cast<const FieldExpr*>(&node))    return lower_field(*e);
        if (auto* e = dynamic_cast<const IndexExpr*>(&node))    return lower_index(*e);
        throw std::runtime_error("IRBuilder: unhandled expression type");
    }
 
    // ── Statement lowering (public for tests) ────────────────────
    void lower_stmt(const ASTNode& node) {
        if (auto* s = dynamic_cast<const LetStmt*>(&node))    { lower_let(*s);       return; }
        if (auto* s = dynamic_cast<const ReturnStmt*>(&node)) { lower_return(*s);    return; }
        if (auto* s = dynamic_cast<const IfStmt*>(&node))     { lower_if(*s);        return; }
        if (auto* s = dynamic_cast<const WhileStmt*>(&node))  { lower_while(*s);     return; }
        if (auto* s = dynamic_cast<const BlockStmt*>(&node))  { lower_block(*s);     return; }
        if (auto* s = dynamic_cast<const ExprStmt*>(&node))   { lower_expr_stmt(*s); return; }
        throw std::runtime_error("IRBuilder: unhandled statement type");
    }
 
private:
    std::unique_ptr<IRModule> mod_;
    Function*   fn_      = nullptr;
    BasicBlock* bb_      = nullptr;
    int         counter_ = 0;
    std::unordered_map<std::string, int> name_counts_;
    std::vector<Scope> scopes_;
    std::vector<BasicBlock*> loop_exit_stack_;
    std::unordered_map<Value*, ValuePtr> ptr_cache_;
 
    // non-owning shared_ptr alias
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
 
    // ── Top-level ────────────────────────────────────────────────
    void lower_top_level(const ASTNode& node) {
        if (auto* d = dynamic_cast<const FnDecl*>(&node))  { lower_fn_decl(*d);    return; }
        if (auto* s = dynamic_cast<const LetStmt*>(&node)) { lower_global_let(*s); return; }
    }
 
    void lower_fn_decl(const FnDecl& d) {
        TypePtr ft = d.resolved_type ? d.resolved_type : Type::fn({}, Type::void_());
        Function* fn = mod_->add_function(d.name, ft, d.is_async);
        fn->is_exported = d.is_exported;
        set_function(fn); push_scope();
        BasicBlock* entry = fn->create_entry(); set_block(entry);
        for (auto& p : d.params) { Argument* a = fn->add_param("%" + p.name, p.type); define(p.name, a); }
        lower_stmts(d.body);
        if (!bb_->is_terminated()) emit<ReturnInst>();
        pop_scope(); set_function(nullptr); set_block(nullptr);
    }
 
    void lower_global_let(const LetStmt& s) {
        auto v = make_const_lit(*s.init); v->name = "@" + s.name; mod_->add_global(std::move(v));
    }
 
    void lower_stmts(const std::vector<std::unique_ptr<ASTNode>>& stmts) {
        for (auto& s : stmts) lower_stmt(*s);
    }
    void lower_block(const BlockStmt& s) { push_scope(); lower_stmts(s.stmts); pop_scope(); }
 
    void lower_let(const LetStmt& s) {
        Value* rhs = lower_expr(*s.init);
        if (s.is_mutable) {
            auto* alloca = emit<AllocaInst>(fresh(s.name + ".slot"), rhs->type);
            emit<StoreInst>(vp(rhs), vp(alloca));
            define(s.name, alloca);
        } else {
            if (rhs->name.empty() || rhs->name[0] == '%') rhs->name = "%" + s.name;
            define(s.name, rhs);
        }
    }
 
    void lower_return(const ReturnStmt& s) {
        if (s.value) emit<ReturnInst>(vp(lower_expr(*s.value)));
        else         emit<ReturnInst>();
    }
 
    void lower_if(const IfStmt& s) {
        Value* cond    = lower_expr(*s.cond);
        auto* then_bb  = fn_->add_block(fresh_label("if.true"));
        auto* else_bb  = s.else_body ? fn_->add_block(fresh_label("if.false")) : nullptr;
        auto* merge_bb = fn_->add_block(fresh_label("if.merge"));
        emit<CondBranchInst>(vp(cond), then_bb, else_bb ? else_bb : merge_bb);
 
        set_block(then_bb); push_scope(); lower_stmts(s.then_body); pop_scope();
        if (!bb_->is_terminated()) emit<BranchInst>(merge_bb);
 
        if (else_bb) {
            set_block(else_bb); push_scope(); lower_stmts(*s.else_body); pop_scope();
            if (!bb_->is_terminated()) emit<BranchInst>(merge_bb);
        }
        set_block(merge_bb);
    }
 
    void lower_while(const WhileStmt& s) {
        auto* header = fn_->add_block(fresh_label("loop.header"));
        auto* body   = fn_->add_block(fresh_label("loop.body"));
        auto* exit   = fn_->add_block(fresh_label("loop.exit"));
        emit<BranchInst>(header);
        set_block(header); Value* cond = lower_expr(*s.cond);
        emit<CondBranchInst>(vp(cond), body, exit);
        loop_exit_stack_.push_back(exit);
        set_block(body); push_scope(); lower_stmts(s.body); pop_scope();
        if (!bb_->is_terminated()) emit<BranchInst>(header);
        loop_exit_stack_.pop_back();
        set_block(exit);
    }
 
    void lower_expr_stmt(const ExprStmt& s) { lower_expr(*s.expr); }
 
    Value* lower_lit_int(const LitIntExpr& e)   { return keep(std::make_shared<ConstantInt>(e.value, e.resolved_type)); }
    Value* lower_lit_float(const LitFloatExpr& e){ return keep(std::make_shared<ConstantFloat>(e.value, e.resolved_type)); }
    Value* lower_lit_bool(const LitBoolExpr& e)  { return keep(std::make_shared<ConstantBool>(e.value)); }
    Value* lower_lit_str(const LitStrExpr& e)    { return keep(std::make_shared<ConstantString>(e.value)); }
 
    Value* lower_ident(const IdentExpr& e) {
        Value* v = lookup(e.name);
        if (!v) throw std::runtime_error("IRBuilder: undefined name '" + e.name + "'");
        if (dynamic_cast<AllocaInst*>(v)) return emit<LoadInst>(fresh(e.name), v->type, vp(v));
        return v;
    }
 
    Value* lower_bin(const BinExpr& e) {
        Value* l = lower_expr(*e.lhs), *r = lower_expr(*e.rhs);
        TypePtr ty = e.resolved_type;
        return emit<BinOpInst>(fresh(), ty, ast_binop(e.op, ty), vp(l), vp(r));
    }
 
    Value* lower_un(const UnExpr& e) {
        Value* o = lower_expr(*e.operand); TypePtr ty = e.resolved_type;
        return emit<UnOpInst>(fresh(), ty, ast_unop(e.op, ty), vp(o));
    }
 
    Value* lower_call(const CallExpr& e) {
        if (auto* f = dynamic_cast<const FieldExpr*>(e.callee.get()))
            return lower_tensor_call(*f, e.args, e.resolved_type);
        Value* callee = lower_expr(*e.callee);
        std::vector<ValuePtr> args;
        for (auto& a : e.args) args.push_back(vp(lower_expr(*a)));
        TypePtr ret = e.resolved_type;
        return emit<CallInst>(ret->is_void() ? "" : fresh(), ret, vp(callee), std::move(args));
    }
 
    Value* lower_spawn(const SpawnExpr& e) {
        return emit<SpawnInst>(fresh("h"), Type::infer(), vp(lower_expr(*e.expr)));
    }
    Value* lower_await(const AwaitExpr& e) {
        return emit<AwaitInst>(fresh(), e.resolved_type, vp(lower_expr(*e.expr)));
    }
    Value* lower_field(const FieldExpr& e) {
        Value* v = lookup(e.module_alias + "::" + e.field);
        if (!v) throw std::runtime_error("IRBuilder: unresolved field '" + e.module_alias + "::" + e.field + "'");
        return v;
    }
    Value* lower_index(const IndexExpr& e) {
        return emit<TensorOpInst>(fresh(), e.resolved_type, TensorOpCode::Select,
            std::vector<ValuePtr>{ vp(lower_expr(*e.base)), vp(lower_expr(*e.index)) });
    }
 
    Value* lower_tensor_call(const FieldExpr& field,
                              const std::vector<std::unique_ptr<ASTNode>>& ast_args,
                              TypePtr ret_type) {
        std::vector<ValuePtr> args;
        for (auto& a : ast_args) args.push_back(vp(lower_expr(*a)));
        TensorOpCode op = resolve_tensor_op(field.field);
        return emit<TensorOpInst>(op_is_void(op) ? "" : fresh(), ret_type, op, std::move(args));
    }
 
    static BinOpCode ast_binop(const std::string& op, const TypePtr& ty) {
        bool f = ty && ty->is_float();
        if (op == "+")  return f ? BinOpCode::FAdd : BinOpCode::Add;
        if (op == "-")  return f ? BinOpCode::FSub : BinOpCode::Sub;
        if (op == "*")  return f ? BinOpCode::FMul : BinOpCode::Mul;
        if (op == "/")  return f ? BinOpCode::FDiv : BinOpCode::Div;
        if (op == "%")  return BinOpCode::Mod;
        if (op == "&")  return BinOpCode::And;
        if (op == "|")  return BinOpCode::Or;
        if (op == "^")  return BinOpCode::Xor;
        if (op == "<<") return BinOpCode::Shl;
        if (op == ">>") return BinOpCode::Shr;
        throw std::runtime_error("IRBuilder: unknown op '" + op + "'");
    }
    static UnOpCode ast_unop(const std::string& op, const TypePtr& ty) {
        bool f = ty && ty->is_float();
        if (op == "-") return f ? UnOpCode::FNeg : UnOpCode::Neg;
        if (op == "!") return UnOpCode::Not;
        throw std::runtime_error("IRBuilder: unknown op '" + op + "'");
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
            {"median",TensorOpCode::Median},{"sum_dim",TensorOpCode::SumDim},
            {"mean_dim",TensorOpCode::MeanDim},{"max_dim",TensorOpCode::MaxDim},
            {"min_dim",TensorOpCode::MinDim},{"argmax",TensorOpCode::ArgMax},
            {"argmin",TensorOpCode::ArgMin},{"exp",TensorOpCode::Exp},
            {"log",TensorOpCode::Log},{"log2",TensorOpCode::Log2},
            {"log1p",TensorOpCode::Log1p},{"sqrt",TensorOpCode::Sqrt},
            {"rsqrt",TensorOpCode::Rsqrt},{"abs",TensorOpCode::Abs},
            {"sign",TensorOpCode::Sign},{"sin",TensorOpCode::Sin},
            {"cos",TensorOpCode::Cos},{"tan",TensorOpCode::Tan},
            {"floor",TensorOpCode::Floor},{"ceil",TensorOpCode::Ceil},
            {"round",TensorOpCode::Round},{"pow",TensorOpCode::Pow},
            {"clamp",TensorOpCode::Clamp},{"lerp",TensorOpCode::Lerp},
            {"relu",TensorOpCode::Relu},{"relu6",TensorOpCode::Relu6},
            {"silu",TensorOpCode::Silu},{"gelu",TensorOpCode::Gelu},
            {"sigmoid",TensorOpCode::Sigmoid},{"tanh",TensorOpCode::Tanh},
            {"softmax",TensorOpCode::Softmax},{"log_softmax",TensorOpCode::LogSoftmax},
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
            {"zero_grad",TensorOpCode::ZeroGrad},{"requires_grad",TensorOpCode::RequiresGrad},
        };
        auto it = T.find(n);
        if (it == T.end()) throw std::runtime_error("IRBuilder: unknown tensor op '" + n + "'");
        return it->second;
    }
 
    static bool op_is_void(TensorOpCode op) {
        return op == TensorOpCode::Backward || op == TensorOpCode::ZeroGrad || op == TensorOpCode::NoGrad;
    }
 
    std::string fresh_label(const std::string& base) {
        auto it = name_counts_.find(base);
        if (it == name_counts_.end()) { name_counts_[base] = 1; return base; }
        return base + std::to_string(it->second++);
    }
 
    std::shared_ptr<Value> make_const_lit(const ASTNode& node) {
        if (auto* e = dynamic_cast<const LitIntExpr*>(&node))   return std::make_shared<ConstantInt>(e->value, e->resolved_type);
        if (auto* e = dynamic_cast<const LitFloatExpr*>(&node)) return std::make_shared<ConstantFloat>(e->value, e->resolved_type);
        if (auto* e = dynamic_cast<const LitBoolExpr*>(&node))  return std::make_shared<ConstantBool>(e->value);
        if (auto* e = dynamic_cast<const LitStrExpr*>(&node))   return std::make_shared<ConstantString>(e->value);
        throw std::runtime_error("IRBuilder: global let must be a literal");
    }
};
 
} // namespace ir