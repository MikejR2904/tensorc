#pragma once
 
/// ir/IRPasses.h
/// Analysis and transformation passes over IRModule.
///
/// Passes (in recommended pipeline order)
/// ────────────────────────────────────────
///   1. TypePropPass      — fill Infer types from operand types
///   2. CFGPass           — populate BasicBlock::preds / succs
///   3. DeadBlockPass     — remove unreachable blocks
///   4. FusionPass        — fuse adjacent compatible TensorOp pairs
///
/// Each pass exposes a static run(IRModule&) method.
/// Passes are idempotent: running twice is safe.
 
#include "IRModule.h"
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
 
namespace ir {
 
// ═════════════════════════════════════════════════════════════════════════════
// 1. TypePropPass — forward type propagation
//
// Fills in Type::Kind::Infer results by inferring from operand types.
// Rules (conservative — never narrows a type already set):
//   BinOpInst   → type of lhs (unless lhs is also Infer, try rhs)
//   UnOpInst    → type of operand
//   CmpInst     → already bool, skip
//   TensorOp    → tensor rules below
//   SpawnInst   → handle type stays Infer (runtime decides)
//   AwaitInst   → type of awaited value (unwrap future)
//   CallInst    → return type from callee fn type
//
// Tensor type rules:
//   element-wise (relu, exp, sin …) → same type as first arg
//   reductions   (sum, mean …)      → scalar element type of first arg
//   creation     (zeros, randn …)   → Tensor<f32> (default)
//   matmul / bmm / dot              → same type as first arg
//   spawn of tensor result          → keep as-is (pass does NOT wrap in future)
// ═════════════════════════════════════════════════════════════════════════════
 
class TypePropPass
{
public:
    static void run(IRModule& mod)
    {
        TypePropPass p;
        // Multiple iterations to handle chains
        for (int iter = 0; iter < 4; ++iter)
            for (auto& fn : mod.functions)
                for (auto& bb : fn->blocks)
                    for (auto& inst : bb->insts)
                        p.infer(*inst);
    }
 
private:
    static bool is_infer(const TypePtr& t) {
        return !t || t->kind == Type::Kind::Infer;
    }
 
    static TypePtr element_type(const TypePtr& t) {
        if (!t) return nullptr;
        if (t->kind == Type::Kind::Tensor) return t->inner_type();
        return t;
    }
 
    static TypePtr make_tensor(const TypePtr& elem) {
        return Type::tensor(elem ? elem : Type::f32());
    }
 
    void infer(Instruction& inst)
    {
        if (!is_infer(inst.type)) return;  // already known
 
        if (auto* i = dynamic_cast<BinOpInst*>(&inst)) {
            TypePtr t = best(i->lhs, i->rhs);
            if (!is_infer(t)) inst.type = t;
        }
        else if (auto* i = dynamic_cast<UnOpInst*>(&inst)) {
            if (!is_infer(i->operand->type)) inst.type = i->operand->type;
        }
        else if (auto* i = dynamic_cast<LoadInst*>(&inst)) {
            if (!is_infer(i->ptr->type)) inst.type = i->ptr->type;
        }
        else if (auto* i = dynamic_cast<TensorOpInst*>(&inst)) {
            infer_tensor(*i);
        }
        else if (auto* i = dynamic_cast<AwaitInst*>(&inst)) {
            // await future<T> → T
            // The task that was spawned has the result type
            if (i->handle && !is_infer(i->handle->type))
                inst.type = i->handle->type;
        }
        else if (auto* i = dynamic_cast<SpawnInst*>(&inst)) {
            // spawn: the handle type mirrors the task result type so await can pick it up
            if (i->task && !is_infer(i->task->type))
                inst.type = i->task->type;
        }
        else if (auto* i = dynamic_cast<CallInst*>(&inst)) {
            if (i->callee && i->callee->type &&
                i->callee->type->kind == Type::Kind::Fn &&
                i->callee->type->ret_type() &&
                !is_infer(i->callee->type->ret_type()))
            {
                inst.type = i->callee->type->ret_type();
            }
        }
    }
 
    void infer_tensor(TensorOpInst& i)
    {
        TypePtr first_arg = i.args.empty() ? nullptr : i.args[0]->type;
 
        switch (i.op) {
        // Element-wise: output matches input tensor type
        case TensorOpCode::Relu:  case TensorOpCode::Relu6: case TensorOpCode::Silu:
        case TensorOpCode::Gelu:  case TensorOpCode::Sigmoid: case TensorOpCode::Tanh:
        case TensorOpCode::Softmax: case TensorOpCode::LogSoftmax:
        case TensorOpCode::Hardsigmoid: case TensorOpCode::Hardswish: case TensorOpCode::Mish:
        case TensorOpCode::LeakyRelu: case TensorOpCode::Elu: case TensorOpCode::Celu:
        case TensorOpCode::Selu: case TensorOpCode::Prelu:
        case TensorOpCode::Exp: case TensorOpCode::Log:  case TensorOpCode::Log2:
        case TensorOpCode::Log1p: case TensorOpCode::Sqrt: case TensorOpCode::Rsqrt:
        case TensorOpCode::Abs:  case TensorOpCode::Sign: case TensorOpCode::Sin:
        case TensorOpCode::Cos:  case TensorOpCode::Tan:  case TensorOpCode::Floor:
        case TensorOpCode::Ceil: case TensorOpCode::Round: case TensorOpCode::Neg:
        case TensorOpCode::Reciprocal: case TensorOpCode::Clamp: case TensorOpCode::Lerp:
        case TensorOpCode::Pow:
        case TensorOpCode::Transpose: case TensorOpCode::Permute: case TensorOpCode::Flatten:
        case TensorOpCode::Squeeze: case TensorOpCode::Unsqueeze:
        case TensorOpCode::Contiguous: case TensorOpCode::Clone:
        case TensorOpCode::Slice: case TensorOpCode::Select:
        case TensorOpCode::Cat:  case TensorOpCode::Stack: case TensorOpCode::Tile:
        case TensorOpCode::Repeat: case TensorOpCode::Pad:
        case TensorOpCode::Detach: case TensorOpCode::NoGrad:
        case TensorOpCode::Reshape: case TensorOpCode::View:
            if (first_arg && !is_infer(first_arg))
                i.type = first_arg;
            break;
 
        // Linear algebra: output tensor matches first arg
        case TensorOpCode::MatMul: case TensorOpCode::Bmm: case TensorOpCode::Outer:
        case TensorOpCode::Cross:  case TensorOpCode::Kron: case TensorOpCode::Dot:
        case TensorOpCode::Inverse: case TensorOpCode::PInverse:
        case TensorOpCode::Solve:  case TensorOpCode::Cholesky:
            if (first_arg && !is_infer(first_arg))
                i.type = first_arg;
            break;
 
        // Reductions → scalar element type
        case TensorOpCode::Sum:  case TensorOpCode::Mean: case TensorOpCode::Max:
        case TensorOpCode::Min:  case TensorOpCode::Prod: case TensorOpCode::Norm:
        case TensorOpCode::Std:  case TensorOpCode::Var:  case TensorOpCode::Median:
        case TensorOpCode::Det:  case TensorOpCode::Trace:
            if (first_arg && !is_infer(first_arg))
                i.type = element_type(first_arg);
            break;
 
        // Dim-preserving reductions → keep tensor type
        case TensorOpCode::SumDim: case TensorOpCode::MeanDim:
        case TensorOpCode::MaxDim: case TensorOpCode::MinDim:
        case TensorOpCode::ArgMax: case TensorOpCode::ArgMin:
        case TensorOpCode::AllDim: case TensorOpCode::AnyDim:
        case TensorOpCode::CumSum: case TensorOpCode::CumProd:
            if (first_arg && !is_infer(first_arg))
                i.type = first_arg;
            break;
 
        // Creation ops → Tensor<f32> by default (shape inference is a later pass)
        case TensorOpCode::Zeros:  case TensorOpCode::Ones:  case TensorOpCode::Full:
        case TensorOpCode::Eye:    case TensorOpCode::Rand:  case TensorOpCode::Randn:
        case TensorOpCode::RandInt: case TensorOpCode::Linspace:
            if (is_infer(i.type)) i.type = make_tensor(Type::f32());
            break;
 
        case TensorOpCode::Arange:
            if (is_infer(i.type)) i.type = Type::array(Type::i64());
            break;
 
        case TensorOpCode::FromList:
            // FromList(ints) → Array; FromList(tensors) → Tensor
            if (!i.args.empty() && i.args[0] && !is_infer(i.args[0]->type)) {
                auto& at = i.args[0]->type;
                if (at->kind == Type::Kind::I32 || at->kind == Type::Kind::I64)
                    i.type = Type::array(at);
                else
                    i.type = make_tensor(element_type(at));
            }
            break;
 
        // Index ops → tensor type
        case TensorOpCode::Gather: case TensorOpCode::TopK:
        case TensorOpCode::Sort:   case TensorOpCode::ArgSort:
        case TensorOpCode::Where:  case TensorOpCode::NonZero:
        case TensorOpCode::MaskedSelect: case TensorOpCode::Scatter:
            if (first_arg && !is_infer(first_arg))
                i.type = first_arg;
            break;
 
        // Autodiff — void ops
        case TensorOpCode::Backward: case TensorOpCode::ZeroGrad:
            i.type = Type::void_();
            break;
 
        case TensorOpCode::Grad: case TensorOpCode::RequiresGrad:
            i.type = Type::bool_();
            break;
 
        default:
            break;
        }
    }
 
    // Pick best known type from two operands
    static TypePtr best(const ValuePtr& a, const ValuePtr& b) {
        if (a && !is_infer(a->type)) return a->type;
        if (b && !is_infer(b->type)) return b->type;
        return Type::infer();
    }
};
 
// ═════════════════════════════════════════════════════════════════════════════
// 2. CFGPass — populate preds / succs on every BasicBlock
// ═════════════════════════════════════════════════════════════════════════════
 
class CFGPass
{
public:
    static void run(IRModule& mod)
    {
        for (auto& fn : mod.functions)
            run_fn(*fn);
    }
 
private:
    static void run_fn(Function& fn)
    {
        // Clear existing edges
        for (auto& bb : fn.blocks) { bb->preds.clear(); bb->succs.clear(); }
 
        for (auto& bb : fn.blocks)
        {
            if (bb->insts.empty()) continue;
            auto* term = bb->insts.back().get();
 
            auto link = [&](BasicBlock* succ) {
                if (!succ) return;
                bb->succs.push_back(succ);
                succ->preds.push_back(bb.get());
            };
 
            if (auto* br = dynamic_cast<BranchInst*>(term))
                link(br->target);
            else if (auto* cbr = dynamic_cast<CondBranchInst*>(term)) {
                link(cbr->true_bb);
                link(cbr->false_bb);
            }
            // ReturnInst → no successors
        }
    }
};
 
// ═════════════════════════════════════════════════════════════════════════════
// 3. DeadBlockPass — remove unreachable basic blocks
//
// Blocks are unreachable if they have no predecessors (except entry).
// Dead blocks caused by if-branches where both arms unconditionally branch
// away also get pruned.
// ═════════════════════════════════════════════════════════════════════════════
 
class DeadBlockPass
{
public:
    static void run(IRModule& mod)
    {
        // Run CFG first to populate preds
        CFGPass::run(mod);
        for (auto& fn : mod.functions)
            run_fn(*fn);
    }
 
private:
    static void run_fn(Function& fn)
    {
        if (fn.blocks.empty()) return;
 
        // Mark reachable via BFS from entry
        std::unordered_set<BasicBlock*> reachable;
        std::vector<BasicBlock*> worklist = { fn.blocks.front().get() };
        while (!worklist.empty()) {
            auto* bb = worklist.back(); worklist.pop_back();
            if (!reachable.insert(bb).second) continue;
            for (auto* s : bb->succs) worklist.push_back(s);
        }
 
        // Erase unreachable blocks
        fn.blocks.erase(
            std::remove_if(fn.blocks.begin(), fn.blocks.end(),
                [&](const std::shared_ptr<BasicBlock>& bb) {
                    return reachable.find(bb.get()) == reachable.end();
                }),
            fn.blocks.end());
    }
};
 
// ═════════════════════════════════════════════════════════════════════════════
// 4. FusionPass — fuse adjacent TensorOp pairs into compound ops
//
// Current fusion rules (single-basic-block, producer-consumer):
//
//   matmul + relu  → fused.matmul_relu      (linear activation)
//   matmul + gelu  → fused.matmul_gelu
//   matmul + silu  → fused.matmul_silu
//   matmul + tanh  → fused.matmul_tanh
//   softmax(log)   → log_softmax            (already a single op, no fusion needed)
//
// The fused op is represented as a TensorOpInst with a new FusedCode stored
// in the first arg's name slot, or (cleaner) as a new opcode.
// For now we tag it by setting the instruction name prefix "fused." and
// replacing op code with the activation op but keeping a fused flag.
//
// We encode fusion by creating a new TensorOpInst with op = Relu (or other
// activation) but with BOTH the matrix args and the activation arg merged,
// and setting the FusedMatMul flag via a naming convention.  The backend
// recognizes "fused.*" names and routes to the CUDA fused kernel.
//
// A cleaner approach would add FusedMatMulRelu etc. to TensorOpCode —
// that's the TODO when the backend lands.
// ═════════════════════════════════════════════════════════════════════════════
 
// Fusion op codes — extend TensorOpCode when backend is ready.
// For now we annotate with a name prefix and a sentinel arg count.
enum class FusedOp { MatMulRelu, MatMulGelu, MatMulSilu, MatMulTanh };
 
struct FusionResult {
    bool        fused = false;
    FusedOp     op    = FusedOp::MatMulRelu;
    std::string name;
};
 
class FusionPass
{
public:
    static int run(IRModule& mod)
    {
        int fused_count = 0;
        for (auto& fn : mod.functions)
            for (auto& bb : fn->blocks)
                fused_count += run_block(*bb);
        return fused_count;
    }
 
private:
    static bool is_activation(TensorOpCode op) {
        switch (op) {
        case TensorOpCode::Relu: case TensorOpCode::Gelu:
        case TensorOpCode::Silu: case TensorOpCode::Tanh:
            return true;
        default: return false;
        }
    }
 
    static std::string fused_name(TensorOpCode act) {
        switch (act) {
        case TensorOpCode::Relu: return "fused.matmul_relu";
        case TensorOpCode::Gelu: return "fused.matmul_gelu";
        case TensorOpCode::Silu: return "fused.matmul_silu";
        case TensorOpCode::Tanh: return "fused.matmul_tanh";
        default: return "fused.matmul_act";
        }
    }
 
    static int run_block(BasicBlock& bb)
    {
        int count = 0;
        auto& insts = bb.insts;
 
        for (size_t i = 0; i + 1 < insts.size(); ++i)
        {
            auto* mm = dynamic_cast<TensorOpInst*>(insts[i].get());
            if (!mm || mm->op != TensorOpCode::MatMul) continue;
            if (mm->name.find("fused.") == 0) continue; // already fused
 
            auto* act = dynamic_cast<TensorOpInst*>(insts[i + 1].get());
            if (!act || !is_activation(act->op)) continue;
 
            // Check producer-consumer: act's first arg must be mm's result
            if (act->args.empty() || act->args[0].get() != mm) continue;
 
            // Fuse: replace mm with fused op, remove act
            // Build fused args: original matmul args (operands of mm)
            std::vector<ValuePtr> fused_args = mm->args;
 
            // Reuse mm's slot with fused naming; result type = act's type
            mm->name = fused_name(act->op) + "." + mm->name.substr(1); // strip %
            mm->type = act->type;
            mm->op   = act->op;   // activation op — backend recognizes fused prefix
            // Note: mm->args stays as matmul args; backend sees name prefix
 
            // Redirect all uses of act to mm
            for (auto& use_bb : bb.parent->blocks)
                for (auto& use_inst : use_bb->insts)
                    redirect_uses(use_inst.get(), act, mm);
 
            // Remove the activation instruction
            insts.erase(insts.begin() + static_cast<ptrdiff_t>(i + 1));
            ++count;
            // Don't advance i — new i+1 might be fuseable too
        }
        return count;
    }
 
    // Redirect all ValuePtr references from old_val to new_val in an instruction
    static void redirect_uses(Instruction* inst, TensorOpInst* old_val, TensorOpInst* new_val)
    {
        auto redir = [&](ValuePtr& vp) {
            if (vp.get() == old_val) {
                // Create non-owning alias to new_val
                vp = std::shared_ptr<Value>(new_val, [](Value*){});
            }
        };
 
        if (auto* b = dynamic_cast<BinOpInst*>(inst))   { redir(b->lhs); redir(b->rhs); }
        if (auto* t = dynamic_cast<TensorOpInst*>(inst)) { for (auto& a : t->args) redir(a); }
        if (auto* c = dynamic_cast<CallInst*>(inst))     { for (auto& a : c->args) redir(a); }
        if (auto* s = dynamic_cast<SpawnInst*>(inst))    { redir(s->task); }
        if (auto* a = dynamic_cast<AwaitInst*>(inst))    { redir(a->handle); }
        if (auto* r = dynamic_cast<ReturnInst*>(inst))   { if (r->val) redir(*r->val); }
        if (auto* c = dynamic_cast<CondBranchInst*>(inst)){ redir(c->cond); }
        if (auto* u = dynamic_cast<UnOpInst*>(inst))     { redir(u->operand); }
        if (auto* l = dynamic_cast<LoadInst*>(inst))     { redir(l->ptr); }
        if (auto* s = dynamic_cast<StoreInst*>(inst))    { redir(s->val); redir(s->ptr); }
    }
};
 
// ═════════════════════════════════════════════════════════════════════════════
// Pipeline helper — run all passes in order
// ═════════════════════════════════════════════════════════════════════════════
 
struct PassPipeline
{
    static int run(IRModule& mod)
    {
        TypePropPass::run(mod);     // 1. fill in Infer types
        CFGPass::run(mod);          // 2. build pred/succ edges
        DeadBlockPass::run(mod);    // 3. prune dead blocks (re-runs CFG internally)
        int fused = FusionPass::run(mod);  // 4. op fusion
        TypePropPass::run(mod);     // 5. re-prop types after fusion
        return fused;
    }
};
 
} // namespace ir