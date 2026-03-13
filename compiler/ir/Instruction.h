#pragma once
 
#include "Value.h"
#include <optional>
#include <string>
#include <vector>
 
namespace ir {
 
struct BasicBlock;  // forward
 
// ─── Instruction base ────────────────────────────────────────────────────────
 
struct Instruction : Value
{
    BasicBlock* parent = nullptr;   ///< Owning basic block (non-owning ptr)
 
    Instruction(std::string name, TypePtr type)
        : Value(std::move(name), std::move(type)) {}
 
    bool is_instruction() const override { return true; }
 
    /// Register this instruction as a user of each operand.
    void track_uses(const std::vector<ValuePtr>& operands)
    {
        for (auto& op : operands) op->add_use(this);
    }
};
 
// ─── Opcodes ─────────────────────────────────────────────────────────────────
 
// ── Binary arithmetic / logic ────────────────────────────────────────────────
enum class BinOpCode {
    Add, Sub, Mul, Div, Mod,        // arithmetic
    And, Or, Xor, Shl, Shr,         // bitwise / logical
    FAdd, FSub, FMul, FDiv,         // float variants
};
 
// ── Comparison ───────────────────────────────────────────────────────────────
enum class CmpCode { Eq, Ne, Lt, Le, Gt, Ge };
 
// ── Unary ────────────────────────────────────────────────────────────────────
enum class UnOpCode { Neg, Not, FNeg };
 
// ── Tensor op family ─────────────────────────────────────────────────────────
/// Named tensor operation — maps 1-to-1 with entries in BuiltinRegistry::tensor.
/// Keeping these as a distinct instruction (not Call) lets the backend pattern-
/// match them to kernel dispatch, fusion, and shape-inference passes without
/// inspecting call targets.
enum class TensorOpCode {
    // Creation
    Zeros, Ones, Full, Eye, Arange, Linspace, Rand, Randn, RandInt, FromList,
    // Shape
    Reshape, View, Flatten, Squeeze, Unsqueeze, Transpose, Permute,
    Contiguous, Clone, Cast,
    // Slice / join
    Slice, Select, Cat, Stack, Split, Chunk, Tile, Repeat, Pad,
    // Reduction (full-collapse → scalar)
    Sum, Mean, Max, Min, Prod, Norm, Std, Var, Median,
    // Reduction (dim-preserving)
    SumDim, MeanDim, MaxDim, MinDim,
    ArgMax, ArgMin, AllDim, AnyDim, CumSum, CumProd,
    // Element-wise math
    Exp, Log, Log2, Log1p, Sqrt, Rsqrt, Abs, Sign,
    Sin, Cos, Tan, Floor, Ceil, Round, Neg, Reciprocal,
    Pow, Clamp, Lerp,
    // Activations
    Relu, Relu6, Silu, Gelu, Sigmoid, Tanh, Softmax,
    LogSoftmax, Hardsigmoid, Hardswish, Mish,
    LeakyRelu, Elu, Celu, Selu, Prelu,
    // Linear algebra
    Dot, MatMul, Bmm, Outer, Cross, Kron,
    Inverse, PInverse, Det, Trace, Diag, Triu, Tril,
    Svd, Eig, Qr, Cholesky, Solve,
    // Sort / index
    Sort, ArgSort, TopK, Gather, Scatter, Where, NonZero, MaskedSelect,
    // Autodiff
    Backward, Grad, NoGrad, Detach, ZeroGrad, RequiresGrad,
};
 
// ── Async opcode ─────────────────────────────────────────────────────────────
enum class AsyncCode { Spawn, Await, ParallelFor, ParallelMap, WaitAll, Barrier };
 
// ─── Concrete instructions ───────────────────────────────────────────────────
 
/// Binary operation:  %result = lhs <op> rhs
struct BinOpInst : Instruction
{
    BinOpCode  op;
    ValuePtr   lhs, rhs;
 
    BinOpInst(std::string name, TypePtr type,
              BinOpCode op, ValuePtr lhs, ValuePtr rhs)
        : Instruction(std::move(name), std::move(type))
        , op(op), lhs(lhs), rhs(rhs)
    { track_uses({lhs, rhs}); }
};
 
/// Unary operation:  %result = <op> operand
struct UnOpInst : Instruction
{
    UnOpCode op;
    ValuePtr operand;
 
    UnOpInst(std::string name, TypePtr type, UnOpCode op, ValuePtr operand)
        : Instruction(std::move(name), std::move(type))
        , op(op), operand(operand)
    { track_uses({operand}); }
};
 
/// Comparison:  %result = lhs <cmp> rhs  → bool
struct CmpInst : Instruction
{
    CmpCode  cmp;
    ValuePtr lhs, rhs;
 
    CmpInst(std::string name, CmpCode cmp, ValuePtr lhs, ValuePtr rhs)
        : Instruction(std::move(name), Type::bool_())
        , cmp(cmp), lhs(lhs), rhs(rhs)
    { track_uses({lhs, rhs}); }
};
 
/// Stack allocation:  %ptr = alloca <type>
struct AllocaInst : Instruction
{
    TypePtr alloc_type;   ///< Type of the allocated slot (not the pointer type)
 
    AllocaInst(std::string name, TypePtr alloc_type)
        : Instruction(std::move(name), alloc_type)   // result type = slot type
        , alloc_type(alloc_type) {}
};
 
/// Load from an alloca or global:  %val = load %ptr
struct LoadInst : Instruction
{
    ValuePtr ptr;
 
    LoadInst(std::string name, TypePtr type, ValuePtr ptr)
        : Instruction(std::move(name), std::move(type)), ptr(ptr)
    { track_uses({ptr}); }
};
 
/// Store to an alloca or global:  store %val -> %ptr  (void result)
struct StoreInst : Instruction
{
    ValuePtr val, ptr;
 
    StoreInst(ValuePtr val, ValuePtr ptr)
        : Instruction("", Type::void_()), val(val), ptr(ptr)
    { track_uses({val, ptr}); }
};
 
/// Unconditional branch:  br <target>
struct BranchInst : Instruction
{
    BasicBlock* target;
 
    explicit BranchInst(BasicBlock* target)
        : Instruction("", Type::void_()), target(target) {}
};
 
/// Conditional branch:  br %cond, <true_bb>, <false_bb>
struct CondBranchInst : Instruction
{
    ValuePtr    cond;
    BasicBlock* true_bb;
    BasicBlock* false_bb;
 
    CondBranchInst(ValuePtr cond, BasicBlock* true_bb, BasicBlock* false_bb)
        : Instruction("", Type::void_())
        , cond(cond), true_bb(true_bb), false_bb(false_bb)
    { track_uses({cond}); }
};
 
/// Return:  ret [%val]
struct ReturnInst : Instruction
{
    std::optional<ValuePtr> val;   ///< nullopt for void returns
 
    explicit ReturnInst(std::optional<ValuePtr> val = std::nullopt)
        : Instruction("", Type::void_()), val(val)
    { if (val) track_uses({*val}); }
};
 
/// Function / builtin call:  %result = call <callee>(%arg0, %arg1, …)
/// Used for user-defined function calls.
/// Built-in tensor ops use TensorOpInst instead.
struct CallInst : Instruction
{
    ValuePtr              callee;
    std::vector<ValuePtr> args;
    bool                  is_tail = false;
 
    CallInst(std::string name, TypePtr ret_type,
             ValuePtr callee, std::vector<ValuePtr> args,
             bool is_tail = false)
        : Instruction(std::move(name), std::move(ret_type))
        , callee(callee), args(std::move(args)), is_tail(is_tail)
    {
        std::vector<ValuePtr> all = {callee};
        all.insert(all.end(), this->args.begin(), this->args.end());
        track_uses(all);
    }
};
 
/// PHI node:  %result = phi [%val_bb0, %val_bb1, …]
/// Used to merge values at join points (needed for loops / if-else).
struct PhiInst : Instruction
{
    /// (incoming value, predecessor block) pairs.
    std::vector<std::pair<ValuePtr, BasicBlock*>> incoming;
 
    PhiInst(std::string name, TypePtr type)
        : Instruction(std::move(name), std::move(type)) {}
 
    void add_incoming(ValuePtr val, BasicBlock* bb)
    {
        incoming.emplace_back(val, bb);
        track_uses({val});
    }
};
 
// ─── Tensor instruction ──────────────────────────────────────────────────────
 
/// First-class tensor operation.
/// %result = tensor.<op>(%arg0, %arg1, …)
///
/// Keeping tensor ops as distinct instructions (not sugar over CallInst) gives
/// the backend direct access for:
///   - Kernel selection & dispatch (CUDA / Metal / CPU)
///   - Op fusion (matmul + relu → fused kernel)
///   - Shape propagation passes
///   - Autodiff graph construction
struct TensorOpInst : Instruction
{
    TensorOpCode          op;
    std::vector<ValuePtr> args;
 
    /// Optional shape annotation set by the shape-propagation pass.
    std::optional<std::vector<int64_t>> inferred_shape;
 
    /// Set by the autodiff pass — true if this op is on the grad-required path.
    bool requires_grad = false;
 
    TensorOpInst(std::string name, TypePtr type,
                 TensorOpCode op, std::vector<ValuePtr> args)
        : Instruction(std::move(name), std::move(type))
        , op(op), args(std::move(args))
    { track_uses(this->args); }
};
 
// ─── Async instructions ──────────────────────────────────────────────────────
 
/// spawn <task_value>  →  %handle
/// Launches a value (closure / fn-ptr) asynchronously.
/// The result is an opaque future handle.
struct SpawnInst : Instruction
{
    ValuePtr task;
 
    SpawnInst(std::string name, TypePtr handle_type, ValuePtr task)
        : Instruction(std::move(name), std::move(handle_type)), task(task)
    { track_uses({task}); }
};
 
/// await %handle  →  %result
/// Suspends the current coroutine frame until the handle resolves.
struct AwaitInst : Instruction
{
    ValuePtr handle;
 
    AwaitInst(std::string name, TypePtr result_type, ValuePtr handle)
        : Instruction(std::move(name), std::move(result_type)), handle(handle)
    { track_uses({handle}); }
};
 
/// parallel_for %n, %body_fn  →  void
/// Emits a parallel loop dispatched across `num_threads()` workers.
struct ParallelForInst : Instruction
{
    ValuePtr n;          ///< iteration count
    ValuePtr body_fn;    ///< fn(i64) -> void
 
    ParallelForInst(ValuePtr n, ValuePtr body_fn)
        : Instruction("", Type::void_()), n(n), body_fn(body_fn)
    { track_uses({n, body_fn}); }
};
 
/// parallel_map %array, %map_fn  →  %result_array
struct ParallelMapInst : Instruction
{
    ValuePtr array;
    ValuePtr map_fn;
 
    ParallelMapInst(std::string name, TypePtr type,
                    ValuePtr array, ValuePtr map_fn)
        : Instruction(std::move(name), std::move(type))
        , array(array), map_fn(map_fn)
    { track_uses({array, map_fn}); }
};
 
/// barrier  →  void   (thread join point)
struct BarrierInst : Instruction
{
    BarrierInst() : Instruction("", Type::void_()) {}
};
 
// ─── Type / shape coercion ───────────────────────────────────────────────────
 
/// %result = cast %val to <type>
struct CastInst : Instruction
{
    ValuePtr src;
    TypePtr  target_type;
 
    CastInst(std::string name, TypePtr target, ValuePtr src)
        : Instruction(std::move(name), target)
        , src(src), target_type(std::move(target))
    { track_uses({src}); }
};
 
/// %result = reshape %tensor, %shape
struct ReshapeInst : Instruction
{
    ValuePtr tensor;
    ValuePtr shape;   ///< a ConstantTensor or array value holding the new dims
 
    ReshapeInst(std::string name, TypePtr type,
                ValuePtr tensor, ValuePtr shape)
        : Instruction(std::move(name), std::move(type))
        , tensor(tensor), shape(shape)
    { track_uses({tensor, shape}); }
};
 
} // namespace ir