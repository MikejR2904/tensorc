#pragma once
 
/// ir/IRPasses.h
/// Analysis and transformation passes over IRModule.
///
/// Architecture
/// ────────────
///   IPass          — abstract base all passes implement
///   IRVerifier     — asserts IR invariants after each pass (debug builds)
///   TypePropPass   — forward type propagation via InstructionVisitor
///   CFGPass        — populate BasicBlock::preds / succs
///   DeadBlockPass  — remove unreachable blocks (requires CFGPass first)
///   FusionPass     — fuse adjacent MatMul+activation pairs into fused opcodes
///   PassManager    — runs passes with a convergence loop and optional logging
///   PassPipeline   — canonical pipeline built on top of PassManager
///
/// Pipeline order (PassPipeline::build encodes this):
///   1. TypePropPass   → fill Infer types
///   2. CFGPass        → build pred/succ edges
///   3. DeadBlockPass  → prune dead blocks (internally requires CFG edges)
///   4. FusionPass     → op fusion (idempotent; only fuses single-use matmuls)
///   5. TypePropPass   → re-propagate types after fusion
///
/// All passes are idempotent: running twice is safe.
/// IRVerifier is a no-op in release builds (NDEBUG defined).
 
#include "IRModule.h"
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include <stdexcept>
#include <iostream>
 
namespace ir {
 
// ─────────────────────────────────────────────────────────────────────────────
// IPass: uniform interface for all compiler passes.
// run() returns true if the pass mutated the module so PassManager knows
// whether to keep iterating toward a fixed point.
// ─────────────────────────────────────────────────────────────────────────────
struct IPass {
    virtual ~IPass() = default;
    /// Run the pass. Returns true iff the module was modified.
    virtual bool run(IRModule& mod) = 0;
    /// Human-readable name used in diagnostics and PassManager logging.
    virtual std::string_view name() const = 0;
};
 
static inline bool types_equal(const TypePtr& a, const TypePtr& b) {
    if (a == b) return true;
    if (!a || !b) return false;
    if (a->kind != b->kind) return false;
    if (a->kind == Type::Kind::Tensor || a->kind == Type::Kind::Array)
        return types_equal(a->inner_type(), b->inner_type());
    return true;
}
 
// ─────────────────────────────────────────────────────────────────────────────
// 1. TypePropPass; forward type propagation
// Implements InstructionVisitor so each instruction type is dispatched without
// dynamic_cast chains. Only overrides visit() for instructions where type
// inference is meaningful; all others are left as the no-op default.
//
// Convergence: iterates until no type changes. A chain of N dependent inferences needs at most N iterations; in practice
// 3–4 is enough for any realistic program, and the changed-flag ensures we
// stop as soon as the IR is stable.
// ─────────────────────────────────────────────────────────────────────────────
class TypePropPass final : public IPass, private InstructionVisitor {
public:
    bool run(IRModule& mod) override {
        bool ever_changed = false;
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& fn : mod.functions) {
                for (auto& bb : fn->blocks) {
                    for (auto& inst : bb->insts) {
                        TypePtr before = inst->type;
                        inst->accept(*this);
                        if (!types_equal(before, inst->type)) {
                            changed = true;
                            ever_changed = true;
                        }
                    }
                }
            }
        }
        return ever_changed;
    }
 
    std::string_view name() const override { return "TypePropPass"; }
 
private:
    // Utilities
    static bool is_infer(const TypePtr& t) { return !t || t->kind == Type::Kind::Infer; }
    /// For Tensor<T>, return T. For a scalar type, return as-is.
    static TypePtr element_type(const TypePtr& t) {
        if (!t) return nullptr;
        return (t->kind == Type::Kind::Tensor) ? t->inner_type() : t;
    }
    static TypePtr make_tensor(const TypePtr& elem) { return Type::tensor(elem ? elem : Type::f32()); }
    /// Pick the first known (non-Infer) type between two operands.
    static TypePtr best(const ValuePtr& a, const ValuePtr& b) {
        if (a && !is_infer(a->type)) return a->type;
        if (b && !is_infer(b->type)) return b->type;
        return nullptr;
    }
    // Visitor overrides
    // Guard: every override returns immediately if the type is already known.
    // We never overwrite a type that has already been resolved.
    void visit(BinOpInst& i) override {
        if (!is_infer(i.type)) return;
        if (auto t = best(i.lhs, i.rhs)) i.type = t;
    }
    void visit(UnOpInst& i) override {
        if (!is_infer(i.type)) return;
        if (i.operand && !is_infer(i.operand->type)) i.type = i.operand->type;
    }
    void visit(CmpInst&) override {
        // CmpInst is always bool — set at construction, nothing to do.
    }
    void visit(LoadInst& i) override {
        if (!is_infer(i.type)) return;
        if (i.ptr && !is_infer(i.ptr->type)) i.type = i.ptr->type;
    }
    void visit(AwaitInst& i) override {
        if (!is_infer(i.type)) return;
        if (i.handle && !is_infer(i.handle->type)) i.type = i.handle->type;
    }
    void visit(SpawnInst& i) override {
        if (!is_infer(i.type)) return;
        if (i.task && !is_infer(i.task->type)) i.type = i.task->type;
    }
    void visit(CallInst& i) override {
        if (!is_infer(i.type)) return;
        if (i.callee && i.callee->type &&
            i.callee->type->kind == Type::Kind::Fn)
        {
            auto ret = i.callee->type->ret_type();
            if (ret && !is_infer(ret)) i.type = ret;
        }
    }
    void visit(TensorOpInst& i) override {
        if (!is_infer(i.type)) return;
        infer_tensor(i);
    }
    // Tensor type inference
    void infer_tensor(TensorOpInst& i) {
        TypePtr first = i.args.empty() ? nullptr : i.args[0]->type;
        switch (i.op) {
        // Element-wise: output == first arg type
        case TensorOpCode::Relu:    case TensorOpCode::Relu6:
        case TensorOpCode::Silu:    case TensorOpCode::Gelu:
        case TensorOpCode::Sigmoid: case TensorOpCode::Tanh:
        case TensorOpCode::Softmax: case TensorOpCode::LogSoftmax:
        case TensorOpCode::Hardsigmoid: case TensorOpCode::Hardswish:
        case TensorOpCode::Mish:    case TensorOpCode::LeakyRelu:
        case TensorOpCode::Elu:     case TensorOpCode::Celu:
        case TensorOpCode::Selu:    case TensorOpCode::Prelu:
        case TensorOpCode::Exp:     case TensorOpCode::Log:
        case TensorOpCode::Log2:    case TensorOpCode::Log1p:
        case TensorOpCode::Sqrt:    case TensorOpCode::Rsqrt:
        case TensorOpCode::Abs:     case TensorOpCode::Sign:
        case TensorOpCode::Sin:     case TensorOpCode::Cos:
        case TensorOpCode::Tan:     case TensorOpCode::Floor:
        case TensorOpCode::Ceil:    case TensorOpCode::Round:
        case TensorOpCode::Neg:     case TensorOpCode::Reciprocal:
        case TensorOpCode::Clamp:   case TensorOpCode::Lerp:
        case TensorOpCode::Pow:
        // Shape-preserving ops
        case TensorOpCode::Transpose: case TensorOpCode::Permute:
        case TensorOpCode::Flatten:   case TensorOpCode::Squeeze:
        case TensorOpCode::Unsqueeze: case TensorOpCode::Contiguous:
        case TensorOpCode::Clone:
        case TensorOpCode::Slice:  case TensorOpCode::Select:
        case TensorOpCode::Cat:    case TensorOpCode::Stack:
        case TensorOpCode::Tile:   case TensorOpCode::Repeat:
        case TensorOpCode::Pad:
        case TensorOpCode::Detach: case TensorOpCode::NoGrad:
        case TensorOpCode::Reshape: case TensorOpCode::View:
            if (first && !is_infer(first)) i.type = first;
            break;
        // Linear algebra: output == first arg type
        case TensorOpCode::MatMul:  case TensorOpCode::Bmm:
        case TensorOpCode::Outer:   case TensorOpCode::Cross:
        case TensorOpCode::Kron:    case TensorOpCode::Dot:
        case TensorOpCode::Inverse: case TensorOpCode::PInverse:
        case TensorOpCode::Solve:   case TensorOpCode::Cholesky:
            if (first && !is_infer(first)) i.type = first;
            break;
        // Full-collapse reductions: Tensor<T> → T (scalar element type)
        case TensorOpCode::Sum:    case TensorOpCode::Mean:
        case TensorOpCode::Max:    case TensorOpCode::Min:
        case TensorOpCode::Prod:   case TensorOpCode::Norm:
        case TensorOpCode::Std:    case TensorOpCode::Var:
        case TensorOpCode::Median: case TensorOpCode::Det:
        case TensorOpCode::Trace:
            if (first && !is_infer(first)) i.type = element_type(first);
            break;
        // Dim-preserving reductions: keep tensor type
        case TensorOpCode::SumDim:  case TensorOpCode::MeanDim:
        case TensorOpCode::MaxDim:  case TensorOpCode::MinDim:
        case TensorOpCode::ArgMax:  case TensorOpCode::ArgMin:
        case TensorOpCode::AllDim:  case TensorOpCode::AnyDim:
        case TensorOpCode::CumSum:  case TensorOpCode::CumProd:
            if (first && !is_infer(first)) i.type = first;
            break;
        // Creation: default to Tensor<f32>
        case TensorOpCode::Zeros:   case TensorOpCode::Ones:
        case TensorOpCode::Full:    case TensorOpCode::Eye:
        case TensorOpCode::Rand:    case TensorOpCode::Randn:
        case TensorOpCode::RandInt: case TensorOpCode::Linspace:
            i.type = make_tensor(Type::f32());
            break;
        case TensorOpCode::Arange:
            i.type = Type::array(Type::i64());
            break;
        case TensorOpCode::FromList:
            // FromList(ints) → Array<T>;  FromList(tensors) → Tensor<T>
            if (first && !is_infer(first)) {
                if (first->kind == Type::Kind::I32 || first->kind == Type::Kind::I64)
                    i.type = Type::array(first);
                else
                    i.type = make_tensor(element_type(first));
            }
            break;
        // Index ops: output == first arg type
        case TensorOpCode::Gather:       case TensorOpCode::TopK:
        case TensorOpCode::Sort:         case TensorOpCode::ArgSort:
        case TensorOpCode::Where:        case TensorOpCode::NonZero:
        case TensorOpCode::MaskedSelect: case TensorOpCode::Scatter:
            if (first && !is_infer(first)) i.type = first;
            break;
        // Autodiff: fixed return types
        case TensorOpCode::Backward: case TensorOpCode::ZeroGrad:
            i.type = Type::void_();
            break;
        case TensorOpCode::Grad: case TensorOpCode::RequiresGrad:
            i.type = Type::bool_();
            break;
        // Fused matmul+activation: inherits matmul output type
        // args still holds the original matmul operands (set by FusionPass).
        case TensorOpCode::FusedMatMulRelu:
        case TensorOpCode::FusedMatMulGelu:
        case TensorOpCode::FusedMatMulSilu:
        case TensorOpCode::FusedMatMulTanh:
            if (first && !is_infer(first)) i.type = first;
            break;
        default:
            break;
        }
    }
};
 
// ─────────────────────────────────────────────────────────────────────────────
// 2. CFGPass; populate preds / succs on every BasicBlock.
// Returns false (no structural change to the IR; edge lists are metadata).
// Clears existing edges before rebuilding so the pass is idempotent.
// ─────────────────────────────────────────────────────────────────────────────
class CFGPass final : public IPass {
public:
    bool run(IRModule& mod) override {
        for (auto& fn : mod.functions) run_fn(*fn);
        // Edge lists are metadata, not IR mutations; callers shouldn't trigger re-convergence just because CFG was rebuilt.
        return false;
    }
    std::string_view name() const override { return "CFGPass"; }
    /// Convenience overload for passes that depend on CFG edges.
    /// Prefer calling through PassManager rather than directly.
    static void run_on(IRModule& mod) {
        for (auto& fn : mod.functions) run_fn(*fn);
    }
 
private:
    static void run_fn(Function& fn) {
        for (auto& bb : fn.blocks) { bb->preds.clear(); bb->succs.clear(); }
 
        for (auto& bb : fn.blocks) {
            if (bb->insts.empty()) continue;
            auto* term = bb->insts.back().get();
            auto link = [&](BasicBlock* succ) {
                if (!succ) return;
                bb->succs.push_back(succ);
                succ->preds.push_back(bb.get());
            };
            if (auto* br  = dynamic_cast<BranchInst*>(term))
                link(br->target);
            else if (auto* cbr = dynamic_cast<CondBranchInst*>(term)) {
                link(cbr->true_bb);
                link(cbr->false_bb);
            }
            // ReturnInst has no successors; no link needed.
        }
    }
};
 
// ─────────────────────────────────────────────────────────────────────────────
// 3. DeadBlockPass; remove unreachable basic blocks via BFS from entry.
//
// Explicit dependency: requires CFG edges to be populated.
// PassPipeline ensures CFGPass runs before this. Running DeadBlockPass
// standalone without first running CFGPass will hit the assert below.
// ─────────────────────────────────────────────────────────────────────────────
class DeadBlockPass final : public IPass {
public:
    bool run(IRModule& mod) override {
        // CFG edges must be current. Rebuild them to guarantee correctness
        // even when called standalone; document this as the explicit dependency.
        CFGPass::run_on(mod);
        bool any_removed = false;
        for (auto& fn : mod.functions)
            if (run_fn(*fn)) any_removed = true;
        return any_removed;
    }
 
    std::string_view name() const override { return "DeadBlockPass"; }
 
private:
    static bool run_fn(Function& fn) {
        if (fn.blocks.empty()) return false;
        // BFS from entry block
        std::unordered_set<BasicBlock*> reachable;
        std::vector<BasicBlock*> worklist = { fn.blocks.front().get() };
        while (!worklist.empty()) {
            auto* bb = worklist.back(); worklist.pop_back();
            if (!reachable.insert(bb).second) continue;
            for (auto* s : bb->succs) worklist.push_back(s);
        }
        const size_t before = fn.blocks.size();
        fn.blocks.erase(
            std::remove_if(fn.blocks.begin(), fn.blocks.end(),
                [&](const std::shared_ptr<BasicBlock>& bb) {
                    return reachable.find(bb.get()) == reachable.end();
                }),
            fn.blocks.end());
        return fn.blocks.size() < before;
    }
};
 
// ─────────────────────────────────────────────────────────────────────────────
// 4. FusionPass; fuse adjacent MatMul → activation into a single fused op.
// Rules
// ─────
//   matmul + relu  →  FusedMatMulRelu
//   matmul + gelu  →  FusedMatMulGelu
//   matmul + silu  →  FusedMatMulSilu
//   matmul + tanh  →  FusedMatMulTanh
// Safety invariants checked before fusing
// ────────────────────────────────────────
//   1. Producer-consumer: act->args[0] must be exactly the matmul instruction.
//   2. Single use: the matmul result must have exactly one user (the activation).
//      If any other instruction also consumes the matmul result, fusing would
//      silently drop a value — so the candidate is skipped.
//   3. Idempotency: already-fused opcodes are never re-fused.
// After fusion
// ────────────
//   - mm->op is set to the appropriate FusedMatMul* opcode.
//   - mm->args retains the original matmul operands — the backend reads these.
//   - The activation instruction is erased from the block.
//   - All former users of the activation are redirected to mm via
//     replaceAllUsesWith, which atomically fixes both use lists.
// Backend contract: is_fused_opcode(op) == true iff the instruction was
// produced by FusionPass.  The backend uses this to route to the fused kernel.
// Only fuse patterns that the backend can actually lower — add new fused
// opcodes only when the matching kernel exists.
// ─────────────────────────────────────────────────────────────────────────────
class FusionPass final : public IPass {
public:
    bool run(IRModule& mod) override {
        int fused = 0;
        for (auto& fn : mod.functions)
            for (auto& bb : fn->blocks)
                fused += fuse_block(*bb);
        last_count_ = fused;
        return fused > 0;
    }
 
    std::string_view name() const override { return "FusionPass"; }
    int fused_count() const { return last_count_; }
 
private:
    static bool is_fusible_activation(TensorOpCode op) {
        switch (op) {
        case TensorOpCode::Relu:
        case TensorOpCode::Gelu:
        case TensorOpCode::Silu:
        case TensorOpCode::Tanh:
            return true;
        default:
            return false;
        }
    }
 
    static TensorOpCode fused_opcode(TensorOpCode act) {
        switch (act) {
        case TensorOpCode::Relu: return TensorOpCode::FusedMatMulRelu;
        case TensorOpCode::Gelu: return TensorOpCode::FusedMatMulGelu;
        case TensorOpCode::Silu: return TensorOpCode::FusedMatMulSilu;
        case TensorOpCode::Tanh: return TensorOpCode::FusedMatMulTanh;
        default:
            // Unreachable: callers guard with is_fusible_activation first.
            assert(false && "fused_opcode called with non-fusible activation");
            return act;
        }
    }
 
    static int fuse_block(BasicBlock& bb) {
        int count = 0;
        auto& insts = bb.insts;
 
        for (size_t i = 0; i + 1 < insts.size(); ++i) {
            // Must be a plain MatMul (not already fused).
            auto* mm = dynamic_cast<TensorOpInst*>(insts[i].get());
            if (!mm || mm->op != TensorOpCode::MatMul) continue;
            if (is_fused_opcode(mm->op)) continue; // idempotency guard
 
            // Next instruction must be a fusible activation.
            auto* act = dynamic_cast<TensorOpInst*>(insts[i + 1].get());
            if (!act || !is_fusible_activation(act->op)) continue;
 
            // Safety check 1: producer-consumer
            // act's first arg must be exactly the matmul result.
            if (act->args.empty() || act->args[0].get() != mm) continue;
            // Safety check 2: single use
            // If the matmul result has any user other than the activation,
            // fusing would remove the intermediate value that those other
            // users depend on.  Skip — do not fuse.
            if (mm->use_count() != 1) continue;
 
            // Fusion transform
            // Mutate mm in place: give it the fused opcode and the activation's result type.  mm->args is unchanged; it still holds the original matmul operands; the backend reads them.
            mm->op = fused_opcode(act->op);
            mm->type = act->type;  // activation's output type becomes mm's type
 
            // Redirect all users of act to mm using the canonical RAUAW utility, which atomically patches use lists on both values.
            ::ir::replaceAllUsesWith(act, insts[i]);
 
            // Remove the now-dead activation instruction.
            insts.erase(insts.begin() + static_cast<ptrdiff_t>(i + 1));
            ++count;
            // Do not advance i: the next instruction (new i+1) may itself be
            // a fusible pair with the freshly-fused op (if extended to chains).
        }
        return count;
    }
private:
    int last_count_ = 0;
};
 
// ─────────────────────────────────────────────────────────────────────────────
// 5. IRVerifier; asserts use-def and structural invariants after each pass.
// Implemented as a pass so PassManager can inject it after every other pass
// in debug builds.  In release builds (NDEBUG), run() is a no-op.
// Invariants checked:
//   A. Every instruction's operand pointer is non-null.
//   B. Every operand's use list contains the consuming instruction.
//   C. Every fused TensorOpInst carries the original matmul args (non-empty).
//   D. Every non-entry BasicBlock has at least one predecessor (post CFGPass).
// Verification failure terminates with an assertion so the error surfaces at
// the pass that produced the invalid IR, not deep in the backend.
// ─────────────────────────────────────────────────────────────────────────────
class IRVerifier final : public IPass, private InstructionVisitor {
public:
    bool run([[maybe_unused]] IRModule& mod) override {
#ifndef NDEBUG
        errors_.clear();
        for (auto& fn : mod.functions) {
            current_fn_ = fn->name;
            for (auto& bb : fn->blocks) {
                current_bb_ = bb->label;
                for (auto& inst : bb->insts) {
                    current_inst_ = inst->name;
                    inst->accept(*this);
                    check_use_def(*inst);
                }
            }
        }
        if (!errors_.empty()) {
            std::cerr << "[IRVerifier] " << errors_.size() << " error(s):\n";
            for (auto& e : errors_) std::cerr << "  " << e << "\n";
            assert(false && "IRVerifier: IR invariants violated (see stderr)");
        }
#endif
        return false; // verifier never mutates
    }
 
    std::string_view name() const override { return "IRVerifier"; }
 
private:
#ifndef NDEBUG
    std::string current_fn_, current_bb_, current_inst_;
    std::vector<std::string> errors_;
 
    void error(const std::string& msg) {
        errors_.push_back(current_fn_ + "::" + current_bb_ + "::" + current_inst_ + ": " + msg);
    }
 
    /// Check that every operand of inst is tracked in operand->uses.
    void check_use_def(const Instruction& inst) {
        auto check_one = [&](const ValuePtr& vp, const char* field) {
            if (!vp) { error(std::string(field) + " operand is null"); return; }
            bool found = std::find(vp->uses.begin(), vp->uses.end(), const_cast<Instruction*>(&inst)) != vp->uses.end();
            if (!found)
                error(std::string(field) + " operand '" + vp->name + "' does not list this instruction as a user");
        };
 
        if (auto* b  = dynamic_cast<const BinOpInst*>(&inst))
            { check_one(b->lhs, "lhs"); check_one(b->rhs, "rhs"); }
        else if (auto* u = dynamic_cast<const UnOpInst*>(&inst))
            { check_one(u->operand, "operand"); }
        else if (auto* c = dynamic_cast<const CmpInst*>(&inst))
            { check_one(c->lhs, "lhs"); check_one(c->rhs, "rhs"); }
        else if (auto* l = dynamic_cast<const LoadInst*>(&inst))
            { check_one(l->ptr, "ptr"); }
        else if (auto* s = dynamic_cast<const StoreInst*>(&inst))
            { check_one(s->val, "val"); check_one(s->ptr, "ptr"); }
        else if (auto* cb = dynamic_cast<const CondBranchInst*>(&inst))
            { check_one(cb->cond, "cond"); }
        else if (auto* r = dynamic_cast<const ReturnInst*>(&inst))
            { if (r->val) check_one(*r->val, "ret_val"); }
        else if (auto* ci = dynamic_cast<const CallInst*>(&inst)) {
            check_one(ci->callee, "callee");
            for (size_t k = 0; k < ci->args.size(); ++k)
                check_one(ci->args[k], ("call_arg[" + std::to_string(k) + "]").c_str());
        }
        else if (auto* ph = dynamic_cast<const PhiInst*>(&inst)) {
            for (size_t k = 0; k < ph->incoming.size(); ++k)
                check_one(ph->incoming[k].first, ("phi_incoming[" + std::to_string(k) + "]").c_str());
        }
        else if (auto* t = dynamic_cast<const TensorOpInst*>(&inst)) {
            // Structural check: fused ops must have args (matmul operands).
            if (is_fused_opcode(t->op) && t->args.empty())
                error("fused TensorOpInst has no args (matmul operands missing)");
            for (size_t k = 0; k < t->args.size(); ++k)
                check_one(t->args[k], ("tensor_arg[" + std::to_string(k) + "]").c_str());
        }
        else if (auto* sp = dynamic_cast<const SpawnInst*>(&inst))
            { check_one(sp->task, "task"); }
        else if (auto* aw = dynamic_cast<const AwaitInst*>(&inst))
            { check_one(aw->handle, "handle"); }
        else if (auto* pf = dynamic_cast<const ParallelForInst*>(&inst))
            { check_one(pf->n, "n"); check_one(pf->body_fn, "body_fn"); }
        else if (auto* pm = dynamic_cast<const ParallelMapInst*>(&inst))
            { check_one(pm->array, "array"); check_one(pm->map_fn, "map_fn"); }
        else if (auto* ca = dynamic_cast<const CastInst*>(&inst))
            { check_one(ca->src, "src"); }
        else if (auto* rs = dynamic_cast<const ReshapeInst*>(&inst))
            { check_one(rs->tensor, "tensor"); check_one(rs->shape, "shape"); }
    }
#endif
};
 
// ─────────────────────────────────────────────────────────────────────────────
// PassManager; runs a sequence of passes with a convergence loop.
// Usage:
//   PassManager pm;
//   pm.add<TypePropPass>();
//   pm.add<CFGPass>();
//   pm.run(mod);               // single pass over each
//   pm.run_to_fixed_point(mod) // repeat until no pass reports changes
// Logging:
//   pm.set_logger([](std::string_view pass, bool changed) { ... });
//   When a logger is set, it is called after every individual pass run.
// Debug verification:
//   pm.enable_verification();  // inserts IRVerifier after every pass
// ─────────────────────────────────────────────────────────────────────────────
class PassManager {
public:
    using Logger = std::function<void(std::string_view pass_name, bool changed)>;
 
    template<typename P, typename... Args>
    void add(Args&&... args) {
        passes_.push_back(std::make_unique<P>(std::forward<Args>(args)...));
    }
 
    void add(std::unique_ptr<IPass> pass) {
        passes_.push_back(std::move(pass));
    }
 
    /// Install a callback invoked after each pass with the pass name and
    /// whether it made changes.  Useful for pipeline diagnostics.
    void set_logger(Logger logger) { logger_ = std::move(logger); }
 
    /// Insert an IRVerifier after every registered pass (debug only).
    /// Call this before run() or run_to_fixed_point().
    void enable_verification() { verify_ = true; }
 
    /// Run each pass once in registration order.
    /// Returns true if any pass made changes.
    bool run(IRModule& mod) {
        bool any = false;
        for (auto& p : passes_) {
            bool changed = p->run(mod);
            if (logger_) logger_(p->name(), changed);
            if (changed) any = true;
#ifndef NDEBUG
            if (verify_) {
                IRVerifier v;
                v.run(mod);
            }
#endif
        }
        return any;
    }
 
    /// Repeat the full pass sequence until no pass reports changes
    /// (fixed-point convergence), or until max_iters is reached.
    /// Returns the number of iterations performed.
    /// Emits a warning to stderr if the limit is reached without convergence.
    int run_to_fixed_point(IRModule& mod, int max_iters = 10) {
        for (int iter = 1; iter <= max_iters; ++iter) {
            bool changed = run(mod);
            if (!changed) return iter;
        }
        std::cerr << "[PassManager] Warning: did not converge after "
                  << max_iters << " iterations ('" << source_hint_ << "')\n";
        return max_iters;
    }
 
    /// Label for warning messages (e.g. source file path).
    void set_source_hint(std::string hint) { source_hint_ = std::move(hint); }
 
private:
    std::vector<std::unique_ptr<IPass>> passes_;
    Logger logger_;
    bool verify_ = false;
    std::string source_hint_;
};
 
// ─────────────────────────────────────────────────────────────────────────────
// PassPipeline; canonical pipeline built on top of PassManager.
// Order rationale:
//   TypeProp first: fills Infer types so CFG and fusion have accurate types.
//   CFG before DeadBlock: DeadBlockPass depends on pred/succ edges.
//   DeadBlock before Fusion: no point fusing ops in dead blocks.
//   Fusion: single pass; idempotent.
//   TypeProp again: fusion may change result types (fused opcode != matmul type).
// To enable pipeline diagnostics pass a logger; to run in debug mode with
// verification after every pass call with verify=true.
// ─────────────────────────────────────────────────────────────────────────────
struct PassPipeline {
    static PassManager build(bool verify = false, PassManager::Logger logger = nullptr)
    {
        PassManager pm;
        if (logger) pm.set_logger(std::move(logger));
        if (verify) pm.enable_verification();
 
        pm.add<TypePropPass>();   // fill Infer types
        pm.add<CFGPass>();        // build CFG edges
        pm.add<DeadBlockPass>();  // prune dead code (re-runs CFG internally)
        pm.add<FusionPass>();     // op fusion
        pm.add<TypePropPass>();   // re-propagate types post-fusion
 
        return pm;
    }
 
    /// Build and run in one call. Returns fused-op count
    /// (requires FusionPass to have been the last mutating pass; use build() + run_to_fixed_point() for full control).
    static int run(IRModule& mod, bool verify = false) {
        TypePropPass tp1;
        tp1.run(mod);
        CFGPass::run_on(mod);
        DeadBlockPass dbp;
        dbp.run(mod);
        FusionPass fusion;
        fusion.run(mod);
        TypePropPass tp2;
        tp2.run(mod);
        return fusion.fused_count();
    }
};
 
} // namespace ir