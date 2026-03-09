#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <cassert>
#include "../lexer/Position.h"
#include "../lexer/TokenKind.h"

struct Expr;
struct Stmt;
struct Compound;
struct Func;
struct MatchArm;

using ExprPtr  = std::unique_ptr<Expr>;
using StmtPtr  = std::unique_ptr<Stmt>;

enum class TyKind
{
    // primitives
    I32, I64,
    F32, F64,
    Bool,
    Str,
    Void,
    // collections
    Tensor,
    Map,
    Set,
    Queue,
    Stack,
    Tuple,
    Array,
    // fn(T) -> T, generic type parameter (name stored separately)
    FnType,
    Generic,
    // type not annotated, to be inferred later
    Infer,
    // named user type  e.g. MyStruct
    UserDef,
};

enum class IdentCtx
{
    Def,        // variable definition
    Ref,        // variable reference
    Param,      // function parameter
    FuncCall,   // invocation
    FuncDef,    // function definition
};

struct IdentInfo
{
    std::string name;
    TyKind      ty_kind;
    IdentCtx    ctx;
    Position    pos;
    IdentInfo(std::string n, TyKind t, IdentCtx c, Position p)
        : name(std::move(n)), ty_kind(t), ctx(c), pos(p) {}
};

struct Ident
{
    enum class Kind { Unqual, Qual } kind;
    IdentInfo info;                         // the leaf name
    std::optional<IdentInfo> qualifier;

    static Ident unqual(IdentInfo info) { return Ident{ Kind::Unqual, std::move(info), std::nullopt }; }
    static Ident qual(IdentInfo qualifier, IdentInfo info) { return Ident{ Kind::Qual, std::move(info), std::move(qualifier) }; }

    const std::string& name()    const { return info.name; }
    TyKind             ty_kind() const { return info.ty_kind; }
    IdentCtx           ctx()     const { return info.ctx; }
    const Position&    pos()     const { return info.pos; }
    void set_ty_kind(TyKind t) { info.ty_kind = t; }
    void set_ctx(IdentCtx c)   { info.ctx = c; }
};

struct Compound
{
    std::vector<StmtPtr>  stmts;
    std::optional<size_t> break_idx;
    Position              pos;
    void addStmt(StmtPtr s)         { stmts.push_back(std::move(s)); }
    void setBreakIdx(size_t i)      { break_idx = i; }
    ExprPtr               tail_expr;
};

enum class BinOp
{
    // arithmetic
    Add, Sub, Mul, Div, MatMul,
    // comparison
    Eq, Neq, Lt, Gt, Lte, Gte,
    // logical
    And, Or,
    // assignment
    Assign, AddAssign, SubAssign, MulAssign, DivAssign,
    // special
    Pipe,           // |>
    Range,          // ..
};

enum class UnaryOp
{
    Neg,    // -x
    Not,    // !x
};

struct GenericParams
{
    std::vector<TyKind>  type_params;   // f32, i32 ...
    std::vector<int>     shape;         // [3, 4]
    Position             pos;
};

struct LitKind
{
    enum class Tag { Int, Float, Str, Bool } tag;
    std::string str_val;
    bool        bool_val = false;

    static LitKind makeInt(std::string v) { return LitKind{ Tag::Int, std::move(v), false }; }
    static LitKind makeFloat(std::string v) { return LitKind{ Tag::Float, std::move(v), false }; }
    static LitKind makeStr(std::string v) { return LitKind{ Tag::Str, std::move(v), false }; }
    static LitKind makeBool(bool v) { return LitKind{ Tag::Bool, "", v }; }
};

struct MatchArm
{
    ExprPtr               pattern;
    std::optional<ExprPtr> guard;
    ExprPtr               body;
    StmtPtr               body_stmt;
    Position              pos;
    bool hasStmtBody() const { return body_stmt != nullptr; }
};

struct ExprKind
{
    enum class Tag
    {
        Lit,
        Id,
        Binary,
        Unary,
        Assign,
        Call,
        Index,          // expr[i]
        Field,          // expr.member
        Scope,          // expr::member
        Range,          // lo..hi
        Pipe,           // lhs |> rhs
        ChannelSend,    // ch <- val
        Await,          // await expr
        Grad,           // grad(loss, params)
        If,             // if as expression
        Match,          // match as expression
        Block,          // { stmts... tail_expr }
        FnExpr,         // fn(params) -> T { }
        // collection literals
        ArrayLit,
        TensorLit,
        SetLit,
        MapLit,
        QueueLit,
        StackLit,
        TupleLit,
    } tag;

    LitKind lit;
    Ident id{ Ident::unqual(IdentInfo{"", TyKind::Infer, IdentCtx::Ref, Position{}}) };
    BinOp   bin_op = BinOp::Add;
    ExprPtr lhs;
    ExprPtr rhs;
    UnaryOp unary_op = UnaryOp::Neg;
    ExprPtr operand;
    ExprPtr                callee;
    std::vector<ExprPtr>   args;
    ExprPtr index;
    ExprPtr     target;
    std::string member;
    ExprPtr condition;
    ExprPtr then_branch;
    ExprPtr else_branch;
    ExprPtr                match_subject;
    std::vector<MatchArm>  arms;
    Compound block;
    std::vector<std::string>               fn_generic_names;
    std::vector<std::pair<std::string, TyKind>> fn_params;
    TyKind                                 fn_ret_type = TyKind::Void;
    Compound                               fn_body;
    ExprPtr pipe_lhs;
    ExprPtr pipe_rhs;
    ExprPtr channel;
    ExprPtr send_val;
    ExprPtr awaited;
    ExprPtr grad_loss;
    ExprPtr grad_params;
    std::vector<ExprPtr>  elements;     // array / set / queue / stack / tuple
    std::vector<std::vector<ExprPtr>> rows;   // tensor  (row-major)
    std::optional<GenericParams> generic_params;
    std::vector<std::pair<ExprPtr,ExprPtr>> map_pairs;  // Map{"k": v}

    Expr* subject() const
    {
        switch (tag)
        {
            case Tag::Index:       return target.get();
            case Tag::Field:       return target.get();
            case Tag::Scope:       return target.get();
            case Tag::Unary:       return operand.get();
            case Tag::Await:       return awaited.get();
            case Tag::ChannelSend: return channel.get();
            default:               return nullptr;
        }
    }

    Expr* value() const
    {
        switch (tag)
        {
            case Tag::Binary:
            case Tag::Assign:      return rhs.get();
            case Tag::Index:       return index.get();      // the subscript
            case Tag::ChannelSend: return send_val.get();   // what is sent
            default:               return nullptr;
        }
    }

    // Returns the string name relevant to this node.
    const std::string& name() const
    {
        // Field / Scope: the member name on the right of '.' or '::'
        if (tag == Tag::Field || tag == Tag::Scope) return member;
        // Id: the identifier string
        if (tag == Tag::Id) return id.name();
        // For everything else this is a programming error.
        static const std::string empty;
        return empty;
    }

    static ExprKind makeLit(LitKind l) {
        ExprKind ek;
        ek.tag = Tag::Lit;
        ek.lit = std::move(l);
        return ek;
    }

    static ExprKind makeId(Ident id) {
        ExprKind ek;
        ek.tag = Tag::Id;
        ek.id = std::move(id);
        return ek;
    }

    static ExprKind makeId(std::string name, TyKind ty = TyKind::Infer, IdentCtx ctx = IdentCtx::Ref, Position p = {}) {
        return makeId(Ident::unqual(IdentInfo(std::move(name), ty, ctx, p)));
    }

    static ExprKind makeBinary(BinOp op, ExprPtr lhs, ExprPtr rhs) {
        ExprKind ek;
        ek.tag = Tag::Binary;
        ek.bin_op = op;
        ek.lhs = std::move(lhs);
        ek.rhs = std::move(rhs);
        return ek;
    }

    static ExprKind makeCall(ExprPtr callee, std::vector<ExprPtr> args) {
        ExprKind ek;
        ek.tag = Tag::Call;
        ek.callee = std::move(callee);
        ek.args = std::move(args);
        return ek;
    }

    static ExprKind makeIndex(ExprPtr collection, ExprPtr subscript)
    {
        ExprKind ek;
        ek.tag    = Tag::Index;
        ek.target = std::move(collection);
        ek.index  = std::move(subscript);
        return ek;
    }

    static ExprKind makeField(ExprPtr object, std::string field_name)
    {
        ExprKind ek;
        ek.tag    = Tag::Field;
        ek.target = std::move(object);
        ek.member = std::move(field_name);
        return ek;
    }

    static ExprKind makeScope(ExprPtr ns, std::string item_name)
    {
        ExprKind ek;
        ek.tag    = Tag::Scope;
        ek.target = std::move(ns);
        ek.member = std::move(item_name);
        return ek;
    }

    static ExprKind makePipe(ExprPtr lhs, ExprPtr rhs) {
        ExprKind ek;
        ek.tag = Tag::Pipe;
        ek.pipe_lhs = std::move(lhs);
        ek.pipe_rhs = std::move(rhs);
        return ek;
    }

    static ExprKind makeTensorLit(GenericParams gp) {
        ExprKind ek;
        ek.tag = Tag::TensorLit;
        ek.generic_params = std::move(gp);
        return ek;
    }

    static ExprKind makeGrad(ExprPtr loss, ExprPtr params) {
        ExprKind ek;
        ek.tag = Tag::Grad;
        ek.grad_loss = std::move(loss);
        ek.grad_params = std::move(params);
        return ek;
    }
};

struct Expr
{
    ExprKind kind;
    Position pos;
    Expr(ExprKind k, Position p)
        : kind(std::move(k)), pos(p) {}
};

struct Func
{
    Ident       ident;      // name + return type stored in ident.ty_kind
    std::vector<std::string> generic_names;
    std::vector<Ident>       params;
    Compound    body;
    Func(Ident id, std::vector<Ident> p, Compound b)
        : ident(std::move(id)), params(std::move(p)), body(std::move(b)) {}
};

struct StmtKind
{
    enum class Tag
    {
        Let,        // let x: T = expr
        Func,       // fn name#(T)(params) -> T { }
        Return,     // return expr
        If,         // if / else if / else
        Else,       // else { }
        While,      // while cond { }
        For,        // for x in iter { }
        Match,      // match x { }
        Import,     // import "path" as alias
        Spawn,      // spawn fn { }
        Compound,   // bare block  { }
        Expr,       // expression statement
        Break,      // break
        Continue,   // continue
    } tag;

    Ident   let_ident{ Ident::unqual(IdentInfo{"", TyKind::Infer, IdentCtx::Def, Position{}}) };
    ExprPtr let_expr;
    std::optional<Func> func;
    ExprPtr ret_expr;       // nullptr = bare return
    ExprPtr  if_cond;
    Compound if_body;
    StmtPtr  else_or_else_if;   // Else or If stmt, nullptr if absent
    Compound else_body;
    ExprPtr  while_cond;    // nullptr = infinite loop
    Compound while_body;
    std::string for_var;
    ExprPtr     for_iter;
    Compound    for_body;
    ExprPtr               match_subject;
    std::vector<MatchArm> match_arms;
    std::string import_path;
    std::string import_alias;   // "" if no alias
    StmtPtr spawn_fn;
    Compound compound;
    ExprPtr expr;

    static StmtKind makeLet(Ident id, ExprPtr expr) {
        StmtKind sk; sk.tag = Tag::Let;
        sk.let_ident = std::move(id);
        sk.let_expr = std::move(expr);
        return sk;
    }

    static StmtKind makeFunc(Func f) {
        StmtKind sk; sk.tag = Tag::Func;
        sk.func = std::move(f);
        return sk;
    }

    static StmtKind makeExpr(ExprPtr e) {
        StmtKind sk; sk.tag = Tag::Expr;
        sk.expr = std::move(e);
        return sk;
    }
    
    static StmtKind makeReturn(ExprPtr e) {
        StmtKind sk; sk.tag = Tag::Return;
        sk.ret_expr = std::move(e);
        return sk;
    }
};

struct Stmt
{
    StmtKind kind;
    Position pos;
    Stmt(StmtKind k, Position p)
        : kind(std::move(k)), pos(p) {}
};

struct Program
{
    std::vector<StmtPtr> stmts;
    void addStmt(StmtPtr s) { stmts.push_back(std::move(s)); }
};