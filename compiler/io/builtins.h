#pragma once
 
#include "../ast/SymbolTable.h"
#include "../ast/Type.h"
 
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
 
namespace io {
 
// ─── Shape helpers ────────────────────────────────────────────────────────────
 
/// A compile-time static shape.  All dims must be > 0.
using Shape = std::vector<int64_t>;
 
/// Reserved sentinel — do NOT use in static-shape signatures.
constexpr int64_t DYNAMIC = -1;
 
/// Convenience: produce a rank-1 shape.
inline Shape shape1(int64_t d0)               { return {d0}; }
/// Convenience: produce a rank-2 shape.
inline Shape shape2(int64_t d0, int64_t d1)   { return {d0, d1}; }
/// Convenience: produce a rank-4 shape (NCHW convention).
inline Shape shape4(int64_t n, int64_t c,
                    int64_t h, int64_t w)      { return {n, c, h, w}; }
 
// ─── ModuleExports ────────────────────────────────────────────────────────────
 
/// All symbols exported by a single built-in (or user) module.
struct ModuleExports
{
    std::string                             path;
    std::unordered_map<std::string, Symbol> symbols;
 
    bool define(Symbol sym)
    {
        auto [_, inserted] = symbols.emplace(sym.name, std::move(sym));
        return inserted;
    }
 
    const Symbol* lookup(const std::string& name) const
    {
        auto it = symbols.find(name);
        return (it != symbols.end()) ? &it->second : nullptr;
    }
 
    std::vector<std::string> exported_names() const
    {
        std::vector<std::string> out;
        out.reserve(symbols.size());
        for (auto& [n, _] : symbols) out.push_back(n);
        return out;
    }
};


class BuiltinRegistry
{
public:
    // ── Module registration ──────────────────────────────────────────────────
 
    void add(const std::string& alias, std::shared_ptr<ModuleExports> mod)
    {
        modules_[alias] = std::move(mod);
    }
 
    bool has_module(const std::string& alias) const
    {
        return modules_.count(alias) > 0;
    }
 
    // ── Symbol lookup ────────────────────────────────────────────────────────
 
    const Symbol* lookup(const std::string& alias,
                         const std::string& symbol_name) const
    {
        auto mit = modules_.find(alias);
        if (mit == modules_.end()) return nullptr;
        return mit->second->lookup(symbol_name);
    }
 
    std::optional<std::vector<std::string>>
    exported_names(const std::string& alias) const
    {
        auto mit = modules_.find(alias);
        if (mit == modules_.end()) return std::nullopt;
        return mit->second->exported_names();
    }
 
    // ── Factory ──────────────────────────────────────────────────────────────
 
    /// Construct a registry pre-populated with every built-in module.
    static BuiltinRegistry with_builtins()
    {
        BuiltinRegistry reg;
        reg.register_std();
        reg.register_math();
        reg.register_tensor();
        reg.register_nn();
        reg.register_optim();
        reg.register_data();
        reg.register_parallel();
        return reg;
    }
 
private:
    std::unordered_map<std::string, std::shared_ptr<ModuleExports>> modules_;
 
    // Shorthand used throughout the register_* helpers.
    static Position P() { return {0, 0}; }
 
    // ── std ──────────────────────────────────────────────────────────────────
 
    void register_std()
    {
        auto m = std::make_shared<ModuleExports>();
        m->path = "std";
        Position p = P();
 
        m->define(Symbol("print",     Type::fn({Type::infer()},                 Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("println",   Type::fn({Type::infer()},                 Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("eprint",    Type::fn({Type::infer()},                 Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("eprintln",  Type::fn({Type::infer()},                 Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("read_line", Type::fn({},                              Type::str_()),  IdentCtx::FuncDef, p));
        m->define(Symbol("assert",    Type::fn({Type::bool_()},                 Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("assert_eq", Type::fn({Type::infer(), Type::infer()},  Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("panic",     Type::fn({Type::str_()},                  Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("exit",      Type::fn({Type::i32()},                   Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("type_of",   Type::fn({Type::infer()},                 Type::str_()),  IdentCtx::FuncDef, p));
        m->define(Symbol("len",       Type::fn({Type::array(Type::infer())},    Type::i64()),   IdentCtx::FuncDef, p));
        m->define(Symbol("range",     Type::fn({Type::i64(), Type::i64()},      Type::array(Type::i64())), IdentCtx::FuncDef, p));
 
        modules_["std"] = std::move(m);
    }
 
    // ── math ─────────────────────────────────────────────────────────────────
 
    void register_math()
    {
        auto m = std::make_shared<ModuleExports>();
        m->path = "math";
        Position p = P();
 
        // Unary scalar ops
        for (const char* name : { "sqrt", "cbrt", "abs", "exp", "exp2",
                                   "log", "log2", "log10",
                                   "sin", "cos", "tan",
                                   "asin", "acos", "atan",
                                   "sinh", "cosh", "tanh",
                                   "floor", "ceil", "round", "trunc",
                                   "sign" })
            m->define(Symbol(name, Type::fn({Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
 
        // Binary scalar ops
        m->define(Symbol("pow",   Type::fn({Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("atan2", Type::fn({Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("hypot", Type::fn({Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("fmod",  Type::fn({Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("max",   Type::fn({Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("min",   Type::fn({Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("clamp", Type::fn({Type::f32(), Type::f32(), Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
 
        // Constants
        m->define(Symbol("pi",      Type::f32(), IdentCtx::Def, p));
        m->define(Symbol("e",       Type::f32(), IdentCtx::Def, p));
        m->define(Symbol("inf",     Type::f32(), IdentCtx::Def, p));
        m->define(Symbol("nan",     Type::f32(), IdentCtx::Def, p));
        m->define(Symbol("epsilon", Type::f32(), IdentCtx::Def, p));   // machine epsilon
 
        modules_["math"] = std::move(m);
    }
 
    // ── tensor ───────────────────────────────────────────────────────────────
 
    void register_tensor()
    {
        auto m  = std::make_shared<ModuleExports>();
        m->path = "tensor";
        Position p  = P();
 
        TypePtr shape_t  = Type::array(Type::infer()); // accepts i32/i64 literals equally
        TypePtr tf       = Type::tensor(Type::f32());
        TypePtr ti       = Type::tensor(Type::infer());
 
        // ── Creation ─────────────────────────────────────────────────────────
        m->define(Symbol("zeros",    Type::fn({shape_t},              tf),  IdentCtx::FuncDef, p));
        m->define(Symbol("ones",     Type::fn({shape_t},              tf),  IdentCtx::FuncDef, p));
        m->define(Symbol("full",     Type::fn({shape_t, Type::f32()}, tf),  IdentCtx::FuncDef, p));
        m->define(Symbol("eye",      Type::fn({shape_t},          tf),  IdentCtx::FuncDef, p));
        m->define(Symbol("arange",   Type::fn({Type::i64(), Type::i64(), Type::i64()}, tf), IdentCtx::FuncDef, p));
        m->define(Symbol("linspace", Type::fn({Type::f32(), Type::f32(), Type::i64()}, tf), IdentCtx::FuncDef, p));
        m->define(Symbol("rand",     Type::fn({shape_t},              tf),  IdentCtx::FuncDef, p));
        m->define(Symbol("randn",    Type::fn({shape_t},              tf),  IdentCtx::FuncDef, p));
        m->define(Symbol("randint",  Type::fn({Type::i64(), Type::i64(), shape_t}, Type::tensor(Type::i64())), IdentCtx::FuncDef, p));
        m->define(Symbol("from_list",Type::fn({Type::array(Type::f32())}, tf), IdentCtx::FuncDef, p));
 
        // ── Shape / layout ───────────────────────────────────────────────────
        m->define(Symbol("shape",     Type::fn({ti},              shape_t),         IdentCtx::FuncDef, p));
        m->define(Symbol("rank",      Type::fn({ti},              Type::i64()),     IdentCtx::FuncDef, p));
        m->define(Symbol("numel",     Type::fn({ti},              Type::i64()),     IdentCtx::FuncDef, p));
        m->define(Symbol("reshape",   Type::fn({ti, shape_t},    ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("view",      Type::fn({ti, shape_t},    ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("flatten",   Type::fn({ti},              ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("squeeze",   Type::fn({ti},              ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("unsqueeze", Type::fn({ti, Type::i64()}, ti),             IdentCtx::FuncDef, p));
        m->define(Symbol("transpose", Type::fn({ti, Type::i64(), Type::i64()}, ti),IdentCtx::FuncDef, p));
        m->define(Symbol("permute",   Type::fn({ti, shape_t},    ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("contiguous",Type::fn({ti},              ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("clone",     Type::fn({ti},              ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("cast",      Type::fn({ti, Type::infer()}, ti),           IdentCtx::FuncDef, p));
 
        // ── Slicing / joining ────────────────────────────────────────────────
        m->define(Symbol("slice",  Type::fn({ti, Type::i64(), Type::i64(), Type::i64()}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("select", Type::fn({ti, Type::i64(), Type::i64()}, ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("cat",    Type::fn({Type::array(ti), Type::i64()}, ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("stack",  Type::fn({Type::array(ti), Type::i64()}, ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("split",  Type::fn({ti, Type::i64(), Type::i64()}, Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("chunk",  Type::fn({ti, Type::i64(), Type::i64()}, Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("tile",   Type::fn({ti, shape_t}, ti),                               IdentCtx::FuncDef, p));
        m->define(Symbol("repeat", Type::fn({ti, shape_t}, ti),                               IdentCtx::FuncDef, p));
        m->define(Symbol("pad",    Type::fn({ti, shape_t, Type::f32()}, ti),                  IdentCtx::FuncDef, p));
 
        // ── Reduction ────────────────────────────────────────────────────────
        // Full-tensor collapse (no dim arg) → bare scalar, usable in comparisons.
        for (const char* name : { "sum", "mean", "max", "min", "prod",
                                   "norm", "std", "var", "median" })
            m->define(Symbol(name, Type::fn({ti}, Type::f32()), IdentCtx::FuncDef, p));
        // Dim-reducing variants → still a tensor (rank drops by 1).
        for (const char* name : { "sum", "mean", "max", "min", "prod",
                                   "norm", "std", "var", "median" })
            m->define(Symbol((std::string(name) + "_dim").c_str(),
                             Type::fn({ti, Type::i64()}, tf), IdentCtx::FuncDef, p));
        m->define(Symbol("argmax",  Type::fn({ti, Type::i64()}, Type::tensor(Type::i64())), IdentCtx::FuncDef, p));
        m->define(Symbol("argmin",  Type::fn({ti, Type::i64()}, Type::tensor(Type::i64())), IdentCtx::FuncDef, p));
        // all/any full-collapse → bare bool for use in if-conditions.
        m->define(Symbol("all",     Type::fn({ti},              Type::bool_()),              IdentCtx::FuncDef, p));
        m->define(Symbol("any",     Type::fn({ti},              Type::bool_()),              IdentCtx::FuncDef, p));
        // all/any along a dim → tensor of bool.
        m->define(Symbol("all_dim", Type::fn({ti, Type::i64()}, Type::tensor(Type::bool_())), IdentCtx::FuncDef, p));
        m->define(Symbol("any_dim", Type::fn({ti, Type::i64()}, Type::tensor(Type::bool_())), IdentCtx::FuncDef, p));
        // prefix-scan ops keep rank → return tensor.
        m->define(Symbol("cumsum",  Type::fn({ti, Type::i64()}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("cumprod", Type::fn({ti, Type::i64()}, ti), IdentCtx::FuncDef, p));
 
        // ── Element-wise maths ───────────────────────────────────────────────
        for (const char* name : { "exp", "log", "log2", "log1p",
                                   "sqrt", "rsqrt", "abs", "sign",
                                   "sin", "cos", "tan",
                                   "floor", "ceil", "round",
                                   "neg", "reciprocal" })
            m->define(Symbol(name, Type::fn({ti}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("pow",    Type::fn({ti, Type::f32()}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("clamp",  Type::fn({ti, Type::f32(), Type::f32()}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("lerp",   Type::fn({ti, ti, Type::f32()}, ti), IdentCtx::FuncDef, p));
 
        // ── Activations ──────────────────────────────────────────────────────
        for (const char* name : { "relu", "relu6", "silu", "gelu",
                                   "sigmoid", "tanh", "softmax",
                                   "log_softmax", "hardsigmoid",
                                   "hardswish", "mish" })
            m->define(Symbol(name, Type::fn({ti}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("leaky_relu", Type::fn({ti, Type::f32()}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("elu",        Type::fn({ti, Type::f32()}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("celu",       Type::fn({ti, Type::f32()}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("selu",       Type::fn({ti}, ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("prelu",      Type::fn({ti, ti}, ti),          IdentCtx::FuncDef, p));
 
        // ── Linear algebra ───────────────────────────────────────────────────
        m->define(Symbol("dot",       Type::fn({ti, ti},              Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("matmul",    Type::fn({ti, ti},              ti),          IdentCtx::FuncDef, p));
        m->define(Symbol("bmm",       Type::fn({ti, ti},              ti),          IdentCtx::FuncDef, p));
        m->define(Symbol("outer",     Type::fn({ti, ti},              ti),          IdentCtx::FuncDef, p));
        m->define(Symbol("cross",     Type::fn({ti, ti, Type::i64()}, ti),          IdentCtx::FuncDef, p));
        m->define(Symbol("kron",      Type::fn({ti, ti},              ti),          IdentCtx::FuncDef, p));
        m->define(Symbol("inverse",   Type::fn({ti},                  ti),          IdentCtx::FuncDef, p));
        m->define(Symbol("pinverse",  Type::fn({ti},                  ti),          IdentCtx::FuncDef, p));
        m->define(Symbol("det",       Type::fn({ti},              Type::f32()),     IdentCtx::FuncDef, p));
        m->define(Symbol("trace",     Type::fn({ti},              Type::f32()),     IdentCtx::FuncDef, p));
        m->define(Symbol("diag",      Type::fn({ti, Type::i64()},     ti),  IdentCtx::FuncDef, p));
        m->define(Symbol("triu",      Type::fn({ti, Type::i64()},     ti),  IdentCtx::FuncDef, p));
        m->define(Symbol("tril",      Type::fn({ti, Type::i64()},     ti),  IdentCtx::FuncDef, p));
        m->define(Symbol("svd",       Type::fn({ti},              Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("eig",       Type::fn({ti},              Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("qr",        Type::fn({ti},              Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("cholesky",  Type::fn({ti},              ti),  IdentCtx::FuncDef, p));
        m->define(Symbol("solve",     Type::fn({ti, ti},          ti),  IdentCtx::FuncDef, p));
 
        // ── Sorting / indexing ───────────────────────────────────────────────
        m->define(Symbol("sort",     Type::fn({ti, Type::i64()}, Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("argsort",  Type::fn({ti, Type::i64()}, Type::tensor(Type::i64())), IdentCtx::FuncDef, p));
        m->define(Symbol("topk",     Type::fn({ti, Type::i64()}, Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("gather",   Type::fn({ti, Type::i64(), Type::tensor(Type::i64())}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("scatter",  Type::fn({ti, Type::i64(), Type::tensor(Type::i64()), ti}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("where",    Type::fn({Type::tensor(Type::bool_()), ti, ti}, ti), IdentCtx::FuncDef, p));
        m->define(Symbol("nonzero",  Type::fn({ti}, Type::tensor(Type::i64())), IdentCtx::FuncDef, p));
        m->define(Symbol("masked_select", Type::fn({ti, Type::tensor(Type::bool_())}, ti), IdentCtx::FuncDef, p));
 
        // ── Autodiff ─────────────────────────────────────────────────────────
        m->define(Symbol("backward",    Type::fn({ti},        Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("grad",        Type::fn({ti},        ti),            IdentCtx::FuncDef, p));
        m->define(Symbol("no_grad",     Type::fn({ti},        ti),            IdentCtx::FuncDef, p));
        m->define(Symbol("detach",      Type::fn({ti},        ti),            IdentCtx::FuncDef, p));
        m->define(Symbol("zero_grad",   Type::fn({ti},        Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("requires_grad", Type::fn({ti, Type::bool_()}, ti),  IdentCtx::FuncDef, p));
 
        modules_["tensor"] = std::move(m);
    }
 
    // ── nn ───────────────────────────────────────────────────────────────────
 
    void register_nn()
    {
        auto m  = std::make_shared<ModuleExports>();
        m->path = "nn";
        Position p  = P();
 
        TypePtr ti  = Type::tensor(Type::infer());
        TypePtr tf  = Type::tensor(Type::f32());
 
        // ── Linear / embedding ───────────────────────────────────────────────
        m->define(Symbol("linear",    Type::fn({Type::infer(), Type::infer(), Type::bool_()}, Type::fn({ti}, ti)),               IdentCtx::FuncDef, p));
        m->define(Symbol("embedding", Type::fn({Type::infer(), Type::infer()},                Type::fn({Type::tensor(Type::infer())}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("bilinear",  Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti, ti}, ti)),           IdentCtx::FuncDef, p));
 
        // ── Convolutions ─────────────────────────────────────────────────────
        // (in_ch, out_ch, kernel, stride, padding)
        m->define(Symbol("conv1d",           Type::fn({Type::infer(), Type::infer(), Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("conv2d",           Type::fn({Type::infer(), Type::infer(), Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("conv3d",           Type::fn({Type::infer(), Type::infer(), Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("conv_transpose1d", Type::fn({Type::infer(), Type::infer(), Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("conv_transpose2d", Type::fn({Type::infer(), Type::infer(), Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("depthwise_conv2d", Type::fn({Type::infer(), Type::infer(), Type::infer()},                              Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
 
        // ── Pooling ──────────────────────────────────────────────────────────
        // (kernel_size, stride, padding)
        m->define(Symbol("max_pool1d",          Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("max_pool2d",          Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("avg_pool1d",          Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("avg_pool2d",          Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("adaptive_avg_pool2d", Type::fn({Type::infer(), Type::infer()},                Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("global_avg_pool",     Type::fn({},                                            Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
 
        // ── Normalisation ────────────────────────────────────────────────────
        m->define(Symbol("batch_norm",    Type::fn({Type::infer()},                          Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("layer_norm",    Type::fn({Type::array(Type::infer())},              Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("group_norm",    Type::fn({Type::infer(), Type::infer()},            Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("instance_norm", Type::fn({Type::infer()},                          Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("rms_norm",      Type::fn({Type::array(Type::infer())},              Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
 
        // ── Dropout / regularisation ─────────────────────────────────────────
        // (p = drop probability)
        m->define(Symbol("dropout",    Type::fn({Type::f32()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("dropout2d",  Type::fn({Type::f32()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("alpha_dropout", Type::fn({Type::f32()}, Type::fn({ti}, ti)), IdentCtx::FuncDef, p));
 
        // ── Recurrent layers ─────────────────────────────────────────────────
        // (input_size, hidden_size, num_layers)
        m->define(Symbol("rnn",  Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti, ti},              Type::array(ti))), IdentCtx::FuncDef, p));
        m->define(Symbol("lstm", Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti, Type::array(ti)}, Type::array(ti))), IdentCtx::FuncDef, p));
        m->define(Symbol("gru",  Type::fn({Type::infer(), Type::infer(), Type::infer()}, Type::fn({ti, ti},              Type::array(ti))), IdentCtx::FuncDef, p));
 
        // ── Attention ────────────────────────────────────────────────────────
        // (embed_dim, num_heads)
        m->define(Symbol("multi_head_attention",          Type::fn({Type::infer(), Type::infer()}, Type::fn({ti, ti, ti}, ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("scaled_dot_product_attention",  Type::fn({ti, ti, ti}, ti),                                          IdentCtx::FuncDef, p));
 
        // ── Loss functions ───────────────────────────────────────────────────
        // All losses collapse to a bare f32 scalar — directly usable in
        // comparisons (loss > threshold) and optimiser step calls.
        m->define(Symbol("mse_loss",          Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("mae_loss",          Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("cross_entropy",     Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("nll_loss",          Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("bce_loss",          Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("bce_with_logits",   Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("hinge_loss",        Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("huber_loss",        Type::fn({ti, ti, Type::f32()},       Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("kl_div",            Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("cosine_similarity", Type::fn({ti, ti},                    Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("triplet_loss",      Type::fn({ti, ti, ti, Type::f32()},   Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("contrastive_loss",  Type::fn({ti, ti, Type::f32()},       Type::f32()), IdentCtx::FuncDef, p));
 
        modules_["nn"] = std::move(m);
    }
 
    // ── optim ────────────────────────────────────────────────────────────────
 
    void register_optim()
    {
        auto m  = std::make_shared<ModuleExports>();
        m->path = "optim";
        Position p  = P();
 
        TypePtr params_t = Type::array(Type::tensor(Type::infer()));
        // All optimiser constructors return a step callable.
        // step accepts a bare f32 loss (matching nn loss return types).
        TypePtr step_fn  = Type::fn({Type::f32()}, Type::void_());
 
        // ── First-order ──────────────────────────────────────────────────────
        // SGD(params, lr, momentum=0, weight_decay=0)
        m->define(Symbol("sgd",     Type::fn({params_t, Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // Adam(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0)
        m->define(Symbol("adam",    Type::fn({params_t, Type::f32(), Type::f32(), Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // AdamW(params, lr, weight_decay)
        m->define(Symbol("adamw",   Type::fn({params_t, Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // RMSProp(params, lr, alpha=0.99, eps=1e-8)
        m->define(Symbol("rmsprop", Type::fn({params_t, Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // Adagrad(params, lr, lr_decay=0, weight_decay=0)
        m->define(Symbol("adagrad", Type::fn({params_t, Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // Adadelta(params, rho=0.9, eps=1e-6, weight_decay=0)
        m->define(Symbol("adadelta",Type::fn({params_t, Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // AdaMax(params, lr, beta1, beta2)
        m->define(Symbol("adamax",  Type::fn({params_t, Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // LBFGS(params, lr, max_iter)
        m->define(Symbol("lbfgs",   Type::fn({params_t, Type::f32(), Type::i64()}, step_fn), IdentCtx::FuncDef, p));
        // Nadam(params, lr, beta1, beta2)
        m->define(Symbol("nadam",   Type::fn({params_t, Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        // RAdam(params, lr, beta1, beta2, weight_decay)
        m->define(Symbol("radam",   Type::fn({params_t, Type::f32(), Type::f32(), Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
 
        // ── Learning-rate schedulers ─────────────────────────────────────────
        // Each scheduler takes an optimiser step-fn and returns an updated step-fn.
        TypePtr sched_fn = Type::fn({step_fn}, step_fn);
        m->define(Symbol("step_lr",        Type::fn({step_fn, Type::i64(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        m->define(Symbol("cosine_annealing",Type::fn({step_fn, Type::i64()}, step_fn),             IdentCtx::FuncDef, p));
        m->define(Symbol("reduce_on_plateau",Type::fn({step_fn, Type::f32(), Type::i64()}, step_fn),IdentCtx::FuncDef, p));
        m->define(Symbol("warmup_cosine",  Type::fn({step_fn, Type::i64(), Type::i64()}, step_fn), IdentCtx::FuncDef, p));
        m->define(Symbol("cyclic_lr",      Type::fn({step_fn, Type::f32(), Type::f32()}, step_fn), IdentCtx::FuncDef, p));
        m->define(Symbol("one_cycle_lr",   Type::fn({step_fn, Type::f32(), Type::i64()}, step_fn), IdentCtx::FuncDef, p));
 
        // ── Gradient utilities ───────────────────────────────────────────────
        m->define(Symbol("clip_grad_norm",  Type::fn({params_t, Type::f32()}, Type::f32()), IdentCtx::FuncDef, p));
        m->define(Symbol("clip_grad_value", Type::fn({params_t, Type::f32()}, Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("zero_grad",       Type::fn({params_t}, Type::void_()), IdentCtx::FuncDef, p));
 
        modules_["optim"] = std::move(m);
    }
 
    // ── data ─────────────────────────────────────────────────────────────────
 
    void register_data()
    {
        auto m  = std::make_shared<ModuleExports>();
        m->path = "data";
        Position p  = P();
 
        TypePtr ti      = Type::tensor(Type::infer());
        TypePtr batch_t = Type::array(ti);             // a batch = list of tensors
        // DataLoader is represented as fn() -> batch (an iterator factory).
        TypePtr loader_t = Type::fn({}, batch_t);
 
        // ── Dataset construction ─────────────────────────────────────────────
        // tensor_dataset([features_tensor, labels_tensor]) -> dataset handle
        TypePtr dataset_t = Type::array(ti);
        m->define(Symbol("tensor_dataset", Type::fn({batch_t}, dataset_t), IdentCtx::FuncDef, p));
        m->define(Symbol("csv_dataset",    Type::fn({Type::str_()}, dataset_t), IdentCtx::FuncDef, p));
        m->define(Symbol("image_dataset",  Type::fn({Type::str_()}, dataset_t), IdentCtx::FuncDef, p));
        m->define(Symbol("hdf5_dataset",   Type::fn({Type::str_(), Type::str_()}, dataset_t), IdentCtx::FuncDef, p));
 
        // ── DataLoader ───────────────────────────────────────────────────────
        m->define(Symbol("dataloader",      Type::fn({dataset_t, Type::infer(), Type::bool_(), Type::infer()}, loader_t), IdentCtx::FuncDef, p));
        m->define(Symbol("next_batch",      Type::fn({loader_t}, batch_t),           IdentCtx::FuncDef, p));
        m->define(Symbol("has_next",        Type::fn({loader_t}, Type::bool_()),     IdentCtx::FuncDef, p));
        m->define(Symbol("reset",           Type::fn({loader_t}, Type::void_()),     IdentCtx::FuncDef, p));
        m->define(Symbol("dataset_len",     Type::fn({dataset_t}, Type::infer()),    IdentCtx::FuncDef, p));
        m->define(Symbol("num_batches",     Type::fn({loader_t},  Type::infer()),    IdentCtx::FuncDef, p));
 
        // ── Transforms ───────────────────────────────────────────────────────
        TypePtr xform_t = Type::fn({ti}, ti);
        m->define(Symbol("normalize",    Type::fn({ti, ti},                              xform_t), IdentCtx::FuncDef, p));
        m->define(Symbol("random_crop",  Type::fn({Type::infer(), Type::infer()},        xform_t), IdentCtx::FuncDef, p));
        m->define(Symbol("random_flip",  Type::fn({Type::f32()},                         xform_t), IdentCtx::FuncDef, p));
        m->define(Symbol("resize",       Type::fn({Type::infer(), Type::infer()},        xform_t), IdentCtx::FuncDef, p));
        m->define(Symbol("to_tensor",    Type::fn({},                                    xform_t), IdentCtx::FuncDef, p));
        m->define(Symbol("compose",      Type::fn({Type::array(xform_t)},               xform_t), IdentCtx::FuncDef, p));
 
        // ── Splitting ────────────────────────────────────────────────────────
        m->define(Symbol("train_test_split", Type::fn({dataset_t, Type::f32()},   Type::array(dataset_t)), IdentCtx::FuncDef, p));
        m->define(Symbol("kfold",            Type::fn({dataset_t, Type::infer()}, Type::array(dataset_t)), IdentCtx::FuncDef, p));
 
        modules_["data"] = std::move(m);
    }
 
    // ── parallel ─────────────────────────────────────────────────────────────
 
    void register_parallel()
    {
        auto m  = std::make_shared<ModuleExports>();
        m->path = "parallel";
        Position p  = P();
 
        TypePtr ti      = Type::tensor(Type::infer());
        // A "work item" is any zero-argument callable returning void.
        TypePtr task_t  = Type::fn({}, Type::void_());
        TypePtr tasks_t = Type::array(task_t);
 
        // ── Thread dispatch ──────────────────────────────────────────────────
        m->define(Symbol("spawn",           Type::fn({task_t},                                                                    Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("parallel_for",    Type::fn({Type::infer(), Type::fn({Type::infer()}, Type::void_())},                   Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("parallel_map",    Type::fn({Type::array(Type::infer()), Type::fn({Type::infer()}, Type::infer())},      Type::array(Type::infer())), IdentCtx::FuncDef, p));
        m->define(Symbol("wait_all",        Type::fn({tasks_t},                                                                   Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("num_threads",     Type::fn({},                                                                          Type::infer()), IdentCtx::FuncDef, p));
        m->define(Symbol("set_num_threads", Type::fn({Type::infer()},                                                             Type::void_()), IdentCtx::FuncDef, p));
 
        // ── Device management ────────────────────────────────────────────────
        // device("cpu" | "cuda:0" | "metal" | …) -> opaque device handle (str tag)
        m->define(Symbol("device",       Type::fn({Type::str_()}, Type::str_()), IdentCtx::FuncDef, p));
        m->define(Symbol("to_device",    Type::fn({ti, Type::str_()}, ti),       IdentCtx::FuncDef, p));
        m->define(Symbol("current_device", Type::fn({}, Type::str_()),           IdentCtx::FuncDef, p));
        m->define(Symbol("num_devices",  Type::fn({}, Type::i64()),              IdentCtx::FuncDef, p));
        m->define(Symbol("is_available", Type::fn({Type::str_()}, Type::bool_()), IdentCtx::FuncDef, p));
        m->define(Symbol("synchronize",  Type::fn({Type::str_()}, Type::void_()), IdentCtx::FuncDef, p));
 
        // ── Data-parallel tensor ops ─────────────────────────────────────────
        // scatter across devices, gather back
        m->define(Symbol("scatter",          Type::fn({ti, Type::array(Type::str_())}, Type::array(ti)), IdentCtx::FuncDef, p));
        m->define(Symbol("gather",           Type::fn({Type::array(ti), Type::str_()}, ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("broadcast",        Type::fn({ti, Type::array(Type::str_())}, Type::array(ti)), IdentCtx::FuncDef, p));
        // all_reduce(tensors, op = "sum" | "mean" | "max") -> tensor
        m->define(Symbol("all_reduce",       Type::fn({Type::array(ti), Type::str_()}, ti),              IdentCtx::FuncDef, p));
        m->define(Symbol("all_gather",       Type::fn({Type::array(ti)},              Type::array(ti)),  IdentCtx::FuncDef, p));
        m->define(Symbol("reduce_scatter",   Type::fn({Type::array(ti), Type::str_()}, Type::array(ti)), IdentCtx::FuncDef, p));
 
        // ── Shared memory / IPC ──────────────────────────────────────────────
        m->define(Symbol("shared_tensor",    Type::fn({ti},                Type::str_()), IdentCtx::FuncDef, p));  // returns shm key
        m->define(Symbol("attach_tensor",    Type::fn({Type::str_()},      ti),           IdentCtx::FuncDef, p));
        m->define(Symbol("detach_shared",    Type::fn({Type::str_()},      Type::void_()), IdentCtx::FuncDef, p));
 
        // ── Barriers / synchronisation primitives ────────────────────────────
        m->define(Symbol("barrier",          Type::fn({},                  Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("lock",             Type::fn({Type::str_()},      Type::void_()), IdentCtx::FuncDef, p));
        m->define(Symbol("unlock",           Type::fn({Type::str_()},      Type::void_()), IdentCtx::FuncDef, p));
 
        modules_["parallel"] = std::move(m);
    }
};
 
} // namespace io