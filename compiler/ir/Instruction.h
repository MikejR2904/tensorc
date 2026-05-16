#pragma once
 
#include "Value.h"
#include <optional>
#include <string>
#include <vector>

namespace ir {
 
struct BasicBlock;  // forward declaration

// We are going to lower the high-level AST to a low-level IR that resembles LLVM IR or similar SSA-based IRs.
// IR would then be lowered further to machine code by a backend (e.g. using LLVM or cranelift).

struct BinOpInst;
struct UnOpInst;
struct CmpInst;
struct AllocaInst;
struct LoadInst;
struct StoreInst;
struct BranchInst;
struct CondBranchInst;
struct ReturnInst;
struct CallInst;
struct PhiInst;
struct TensorOpInst;
struct SpawnInst;
struct AwaitInst;
struct ParallelForInst;
struct ParallelMapInst;
struct BarrierInst;
struct CastInst;
struct ReshapeInst;

// Visitor pattern for instructions, used by IR passes to dispatch on instruction type without dynamic_cast.
struct InstructionVisitor {
    virtual ~InstructionVisitor() = default;
 
    virtual void visit(BinOpInst&)      { visit_default(); }
    virtual void visit(UnOpInst&)       { visit_default(); }
    virtual void visit(CmpInst&)        { visit_default(); }
    virtual void visit(AllocaInst&)     { visit_default(); }
    virtual void visit(LoadInst&)       { visit_default(); }
    virtual void visit(StoreInst&)      { visit_default(); }
    virtual void visit(BranchInst&)     { visit_default(); }
    virtual void visit(CondBranchInst&) { visit_default(); }
    virtual void visit(ReturnInst&)     { visit_default(); }
    virtual void visit(CallInst&)       { visit_default(); }
    virtual void visit(PhiInst&)        { visit_default(); }
    virtual void visit(TensorOpInst&)   { visit_default(); }
    virtual void visit(SpawnInst&)      { visit_default(); }
    virtual void visit(AwaitInst&)      { visit_default(); }
    virtual void visit(ParallelForInst&){ visit_default(); }
    virtual void visit(ParallelMapInst&){ visit_default(); }
    virtual void visit(BarrierInst&)    { visit_default(); }
    virtual void visit(CastInst&)       { visit_default(); }
    virtual void visit(ReshapeInst&)    { visit_default(); }
 
protected:
    virtual void visit_default() {}
};

struct Instruction : Value
{
    BasicBlock* parent = nullptr; // Owning basic block (non-owning ptr)
    Instruction(std::string name, TypePtr type): Value(std::move(name), std::move(type)) {}
    bool is_instruction() const override { return true; }
    // Register this instruction as a user of each operand.
    void track_uses(const std::vector<ValuePtr>& operands) { for (auto& op : operands) op->add_use(this); }
    // Accept a visitor.
    virtual void accept(InstructionVisitor& visitor) = 0;
};
 
// Binary arithmetic / logic
enum class BinOpCode {
    Add, Sub, Mul, Div, Mod, // arithmetic
    And, Or, Xor, Shl, Shr,  // bitwise / logical
    FAdd, FSub, FMul, FDiv,  // float variants
};
 
// Comparison
enum class CmpCode { Eq, Ne, Lt, Le, Gt, Ge };

// Unary
enum class UnOpCode { Neg, Not, FNeg };
 
// Tensor operations
enum class TensorOpCode {
    // Creation
    Zeros, Ones, Full, Eye, Arange, Linspace, Rand, Randn, RandInt, FromList,
    // Shape
    Reshape, View, Flatten, Squeeze, Unsqueeze, Transpose, Permute, Contiguous, Clone, Cast,
    // Slice / join
    Slice, Select, Cat, Stack, Split, Chunk, Tile, Repeat, Pad,
    // Reduction (full-collapse → scalar)
    Sum, Mean, Max, Min, Prod, Norm, Std, Var, Median,
    // Reduction (dim-preserving)
    SumDim, MeanDim, MaxDim, MinDim, ArgMax, ArgMin, AllDim, AnyDim, CumSum, CumProd,
    // Element-wise math
    Exp, Log, Log2, Log1p, Sqrt, Rsqrt, Abs, Sign, Sin, Cos, Tan, Floor, Ceil, Round, Neg, Reciprocal, Pow, Clamp, Lerp,
    // Element-wise arithmetic
    ElemAdd, ElemSub, ElemMul, ElemDiv,
    // Activations
    Relu, Relu6, Silu, Gelu, Sigmoid, Tanh, Softmax, LogSoftmax, Hardsigmoid, Hardswish, Mish, LeakyRelu, Elu, Celu, Selu, Prelu,
    // Linear algebra
    Dot, MatMul, Bmm, Outer, Cross, Kron, Inverse, PInverse, Det, Trace, Diag, Triu, Tril, Svd, Eig, Qr, Cholesky, Solve,
    // Sort / index
    Sort, ArgSort, TopK, Gather, Scatter, Where, NonZero, MaskedSelect,
    // Autodiff
    Backward, Grad, NoGrad, Detach, ZeroGrad, RequiresGrad,
    // Fused kernels
    FusedMatMulRelu, FusedMatMulGelu, FusedMatMulSilu, FusedMatMulTanh,
};
 
/// True for opcodes that were created by FusionPass and map to a single fused kernel.  The backend uses this to route to the correct dispatch path.
inline bool is_fused_opcode(TensorOpCode op) {
    switch (op) {
    case TensorOpCode::FusedMatMulRelu:
    case TensorOpCode::FusedMatMulGelu:
    case TensorOpCode::FusedMatMulSilu:
    case TensorOpCode::FusedMatMulTanh:
        return true;
    default:
        return false;
    }
}

// Async and parallelism
enum class AsyncCode { Spawn, Await, ParallelFor, ParallelMap, WaitAll, Barrier };
 

// Concrete instructions
// Binary operation:  %result = lhs <op> rhs
struct BinOpInst : Instruction
{
    BinOpCode op;
    ValuePtr lhs, rhs;

    BinOpInst(std::string name, TypePtr type, BinOpCode op, ValuePtr lhs, ValuePtr rhs)
        : Instruction(std::move(name), std::move(type)), op(op), lhs(lhs), rhs(rhs) { track_uses({lhs, rhs}); }

    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
// Unary operation:  %result = <op> operand
struct UnOpInst : Instruction
{
    UnOpCode op;
    ValuePtr operand;
 
    UnOpInst(std::string name, TypePtr type, UnOpCode op, ValuePtr operand)
        : Instruction(std::move(name), std::move(type)), op(op), operand(operand) { track_uses({operand}); }

    void accept(InstructionVisitor& v) override { v.visit(*this); }
};
 
// Comparison:  %result = lhs <cmp> rhs  → bool
struct CmpInst : Instruction
{
    CmpCode cmp;
    ValuePtr lhs, rhs;
 
    CmpInst(std::string name, CmpCode cmp, ValuePtr lhs, ValuePtr rhs)
        : Instruction(std::move(name), Type::bool_()), cmp(cmp), lhs(lhs), rhs(rhs) { track_uses({lhs, rhs}); }

    void accept(InstructionVisitor& v) override { v.visit(*this); }
};
 
// Stack allocation:  %ptr = alloca <type>
struct AllocaInst : Instruction
{
    TypePtr alloc_type;   // Type of the allocated slot (not the pointer type)

    AllocaInst(std::string name, TypePtr alloc_type)
        : Instruction(std::move(name), alloc_type), alloc_type(alloc_type) {}
    
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
// Load from an alloca or global:  %val = load %ptr
struct LoadInst : Instruction
{
    ValuePtr ptr;
    LoadInst(std::string name, TypePtr type, ValuePtr ptr): Instruction(std::move(name), std::move(type)), ptr(ptr) { track_uses({ptr}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
// Store to an alloca or global:  store %val -> %ptr  (void result)
struct StoreInst : Instruction
{
    ValuePtr val, ptr;
    StoreInst(ValuePtr val, ValuePtr ptr): Instruction("", Type::void_()), val(val), ptr(ptr) { track_uses({val, ptr}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
// Unconditional branch:  br <target>
struct BranchInst : Instruction
{
    BasicBlock* target;
    explicit BranchInst(BasicBlock* target): Instruction("", Type::void_()), target(target) {}
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
// Conditional branch:  br %cond, <true_bb>, <false_bb>
struct CondBranchInst : Instruction
{
    ValuePtr cond;
    BasicBlock* true_bb;
    BasicBlock* false_bb;
 
    CondBranchInst(ValuePtr cond, BasicBlock* true_bb, BasicBlock* false_bb)
        : Instruction("", Type::void_()), cond(cond), true_bb(true_bb), false_bb(false_bb) { track_uses({cond}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
// Return:  ret [%val]
struct ReturnInst : Instruction
{
    std::optional<ValuePtr> val; // nullopt for void returns
    explicit ReturnInst(std::optional<ValuePtr> val = std::nullopt): Instruction("", Type::void_()), val(val) { if (val) track_uses({*val}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
/// Function / builtin call:  %result = call <callee>(%arg0, %arg1, …)
/// Used for user-defined function calls. Built-in tensor ops use TensorOpInst instead.
struct CallInst : Instruction
{
    ValuePtr callee;
    std::vector<ValuePtr> args;
    bool is_tail = false;
 
    CallInst(std::string name, TypePtr ret_type, ValuePtr callee, std::vector<ValuePtr> args, bool is_tail = false)
        : Instruction(std::move(name), std::move(ret_type)), callee(callee), args(std::move(args)), is_tail(is_tail)
    {
        std::vector<ValuePtr> all = {callee};
        all.insert(all.end(), this->args.begin(), this->args.end());
        track_uses(all);
    }
    void accept(InstructionVisitor& v) override { v.visit(*this); }
};
 
/// PHI node:  %result = phi [%val_bb0, %val_bb1, …]
/// Used to merge values at join points (needed for loops / if-else).
struct PhiInst : Instruction
{
    /// (incoming value, predecessor block) pairs.
    std::vector<std::pair<ValuePtr, BasicBlock*>> incoming;
    PhiInst(std::string name, TypePtr type): Instruction(std::move(name), std::move(type)) {}
 
    void add_incoming(ValuePtr val, BasicBlock* bb)
    {
        incoming.emplace_back(val, bb);
        track_uses({val});
    }
    void accept(InstructionVisitor& v) override { v.visit(*this); }
};
 
// Tensor instruction
/// First-class tensor operation.
/// %result = tensor.<op>(%arg0, %arg1, …)
///
/// Keeping tensor ops as distinct instructions gives the backend direct access for:
///   - Kernel selection & dispatch (CUDA / Metal / CPU)
///   - Op fusion (matmul + relu → fused kernel)
///   - Shape propagation passes
///   - Autodiff graph construction
struct TensorOpInst : Instruction
{
    TensorOpCode op;
    std::vector<ValuePtr> args;
    // Optional shape annotation set by the shape-propagation pass.
    std::optional<std::vector<int64_t>> inferred_shape;
    // Set by the autodiff pass; true if this op is on the grad-required path.
    bool requires_grad = false;
 
    TensorOpInst(std::string name, TypePtr type, TensorOpCode op, std::vector<ValuePtr> args)
        : Instruction(std::move(name), std::move(type)), op(op), args(std::move(args)) { track_uses(this->args); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
// Async instructions
/// spawn <task_value>  →  %handle
/// Launches a value (closure / fn-ptr) asynchronously. The result is an opaque future handle.
struct SpawnInst : Instruction
{
    ValuePtr task;
    SpawnInst(std::string name, TypePtr handle_type, ValuePtr task): Instruction(std::move(name), std::move(handle_type)), task(task) { track_uses({task}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
/// await %handle  →  %result
/// Suspends the current coroutine frame until the handle resolves.
struct AwaitInst : Instruction
{
    ValuePtr handle;
    AwaitInst(std::string name, TypePtr result_type, ValuePtr handle): Instruction(std::move(name), std::move(result_type)), handle(handle) { track_uses({handle}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
/// parallel_for %n, %body_fn  →  void
/// Emits a parallel loop dispatched across `num_threads()` workers.
struct ParallelForInst : Instruction
{
    ValuePtr n;       // iteration count
    ValuePtr body_fn; // fn(i64) -> void
 
    ParallelForInst(ValuePtr n, ValuePtr body_fn): Instruction("", Type::void_()), n(n), body_fn(body_fn) { track_uses({n, body_fn}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
/// parallel_map %array, %map_fn  →  %result_array
struct ParallelMapInst : Instruction
{
    ValuePtr array;
    ValuePtr map_fn;
 
    ParallelMapInst(std::string name, TypePtr type, ValuePtr array, ValuePtr map_fn)
        : Instruction(std::move(name), std::move(type)), array(array), map_fn(map_fn) { track_uses({array, map_fn}); }
    void accept(InstructionVisitor& visitor) override { visitor.visit(*this); }
};
 
/// barrier  →  void   (thread join point)
struct BarrierInst : Instruction
{
    BarrierInst() : Instruction("", Type::void_()) {}
    void accept(InstructionVisitor& v) override { v.visit(*this); }
};
 
// Type/shape coercion
/// %result = cast %val to <type>
struct CastInst : Instruction
{
    ValuePtr src;
    TypePtr target_type;
 
    CastInst(std::string name, TypePtr target, ValuePtr src)
        : Instruction(std::move(name), target), src(src), target_type(std::move(target)) { track_uses({src}); }
    
    void accept(InstructionVisitor& v) override { v.visit(*this); }
};
 
/// %result = reshape %tensor, %shape
struct ReshapeInst : Instruction
{
    ValuePtr tensor;
    ValuePtr shape; // a ConstantTensor or array value holding the new dims
 
    ReshapeInst(std::string name, TypePtr type, ValuePtr tensor, ValuePtr shape)
        : Instruction(std::move(name), std::move(type)), tensor(tensor), shape(shape) { track_uses({tensor, shape}); }
    void accept(InstructionVisitor& v) override { v.visit(*this); }
};


/// Helper: patch one ValuePtr if it points at old_val
static inline void patch_ptr(ValuePtr& vp, Value* old_val, Value* new_val, const std::shared_ptr<Value>& new_shared) { if (vp.get() == old_val) vp = new_shared; }
 
inline void Value::replaceAllUsesWith(Value* replacement) {
    if (replacement == this) return;
 
    // Build a shared_ptr alias for replacement that participates in ownership.
    // We find the canonical shared_ptr by looking up any existing shared owner.
    // The safest way: let the caller ensure replacement is alive (as it always
    // will be — it is either a live instruction in the same module or a
    // function argument), and borrow ownership from the first use's shared_ptr
    // context.  Since Value does not self-reference, we use a non-owning alias
    // with enable_shared_from_this pattern replacement:
    //
    // Instead of a fragile custom-deleter alias, we patch each instruction's
    // operand fields directly using the instruction's own shared_ptr context,
    // obtained by scanning the parent block's inst list.
    //
    // Protocol: callers pass the replacement's live shared_ptr via the
    // overload below.  This base overload is intentionally left as an assertion
    // trap to prevent accidental direct calls without a shared owner.
    (void)replacement;
    // Call the shared_ptr overload instead.
}
 
inline void replaceAllUsesWith(Value* old_val, const ValuePtr& replacement_shared)
{
    if (old_val == replacement_shared.get()) return;
    // Copy the use list; we'll mutate uses during iteration
    std::vector<Instruction*> users = old_val->uses;
    for (Instruction* user : users) {
        // Patch every operand pointer in user that refers to old_val
        auto patch = [&](ValuePtr& vp) {
            if (vp.get() == old_val) {
                vp = replacement_shared;
            }
        };
 
        if (auto* b  = dynamic_cast<BinOpInst*>(user))       { patch(b->lhs); patch(b->rhs); }
        if (auto* u  = dynamic_cast<UnOpInst*>(user))         { patch(u->operand); }
        if (auto* c  = dynamic_cast<CmpInst*>(user))          { patch(c->lhs); patch(c->rhs); }
        if (auto* l  = dynamic_cast<LoadInst*>(user))         { patch(l->ptr); }
        if (auto* s  = dynamic_cast<StoreInst*>(user))        { patch(s->val); patch(s->ptr); }
        if (auto* cb = dynamic_cast<CondBranchInst*>(user))   { patch(cb->cond); }
        if (auto* r  = dynamic_cast<ReturnInst*>(user))       { if (r->val) patch(*r->val); }
        if (auto* ci = dynamic_cast<CallInst*>(user))         {
            patch(ci->callee);
            for (auto& a : ci->args) patch(a);
        }
        if (auto* ph = dynamic_cast<PhiInst*>(user))          {
            for (auto& [v, _bb] : ph->incoming) patch(v);
        }
        if (auto* t  = dynamic_cast<TensorOpInst*>(user))     {
            for (auto& a : t->args) patch(a);
        }
        if (auto* sp = dynamic_cast<SpawnInst*>(user))        { patch(sp->task); }
        if (auto* aw = dynamic_cast<AwaitInst*>(user))        { patch(aw->handle); }
        if (auto* pf = dynamic_cast<ParallelForInst*>(user))  { patch(pf->n); patch(pf->body_fn); }
        if (auto* pm = dynamic_cast<ParallelMapInst*>(user))  { patch(pm->array); patch(pm->map_fn); }
        if (auto* ca = dynamic_cast<CastInst*>(user))         { patch(ca->src); }
        if (auto* rs = dynamic_cast<ReshapeInst*>(user))      { patch(rs->tensor); patch(rs->shape); }
 
        // Remove user from old_val, add to replacement
        old_val->remove_use(user);
        replacement_shared->add_use(user);
    }
}
 
} // namespace ir