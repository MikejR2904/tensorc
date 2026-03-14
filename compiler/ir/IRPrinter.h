#pragma once
 
/// ir/IRPrinter.h
/// Converts an IRModule (or any sub-node) into human-readable text.
///
/// Output format (TensorC IR text)
/// ────────────────────────────────
///   module "path/to/file.tcc"
///
///   fn @add(%a: f32, %b: f32) -> f32:
///   entry:
///     %sum = fadd %a, %b          ; f32
///     ret %sum
///
///   async fn @forward(%x: Tensor<f32>, %w: Tensor<f32>) -> Tensor<f32>:
///   entry:
///     %mm  = tensor.matmul %x, %w ; Tensor<f32>
///     %h   = spawn %mm            ; _
///     %r   = await %h             ; Tensor<f32>
///     ret %r
///
/// Design notes
/// ────────────
///   - Every result-producing instruction prints:
///       <indent>%name = <mnemonic> <operands>  ; <type>
///   - Void instructions (store, br, ret void, barrier) print without lhs:
///       <indent><mnemonic> <operands>
///   - Type comments ("; <type>") help humans cross-check pass output.
///   - The printer never modifies the IR — purely read-only.
///   - IRPrinter::print(module) → std::string  (main entry point)
///   - Individual print_* methods are public so passes can dump sub-trees.
 
#include "IRModule.h"
#include "../ast/Type.h"
#include <sstream>
#include <string>
#include <unordered_map>
 
namespace ir {
 
class IRPrinter
{
public:
    // ── Main entry points ─────────────────────────────────────────────────────
 
    /// Print an entire module to a string.
    static std::string print(const IRModule& mod)
    {
        IRPrinter p;
        p.emit_module(mod);
        return p.out_.str();
    }
 
    /// Print a single function.
    static std::string print(const Function& fn)
    {
        IRPrinter p;
        p.emit_function(fn);
        return p.out_.str();
    }
 
    /// Print a single basic block.
    static std::string print(const BasicBlock& bb)
    {
        IRPrinter p;
        p.emit_block(bb);
        return p.out_.str();
    }
 
    /// Print a single instruction.
    static std::string print(const Instruction& inst)
    {
        IRPrinter p;
        p.emit_inst(inst);
        return p.out_.str();
    }
 
    // ── Streaming overload — write directly to an ostream ────────────────────
    static void print(const IRModule& mod, std::ostream& os)
    {
        os << print(mod);
    }
 
private:
    std::ostringstream out_;
    int                indent_ = 0;   ///< current indent level (steps of 2)
 
    // ── Indentation helpers ───────────────────────────────────────────────────
 
    void push() { indent_ += 2; }
    void pop()  { indent_ -= 2; }
 
    void line(const std::string& s = "")
    {
        if (!s.empty())
            out_ << std::string(indent_, ' ') << s << "\n";
        else
            out_ << "\n";
    }
 
    // ── Module ────────────────────────────────────────────────────────────────
 
    void emit_module(const IRModule& mod)
    {
        out_ << "module \"" << mod.source_path << "\"\n";
 
        // Imported module paths
        for (auto* imp : mod.imports)
            out_ << "import \"" << imp->source_path << "\"\n";
 
        if (!mod.imports.empty() || !mod.globals.empty())
            out_ << "\n";
 
        // Global constants / let bindings
        for (auto& g : mod.globals)
        {
            const std::string& gname = g->name;
            out_ << ((!gname.empty() && gname[0] == '@') ? gname : "@" + gname);
            out_ << " = ";
            emit_constant_value(*g);
            out_ << "  ; " << type_str(g->type) << "\n";
        }
 
        if (!mod.globals.empty()) out_ << "\n";
 
        // Functions (separated by blank lines)
        for (size_t i = 0; i < mod.functions.size(); ++i)
        {
            const auto& fn = *mod.functions[i];
            if (fn.blocks.empty()) continue;
            emit_function(fn);
            if (i + 1 < mod.functions.size()) out_ << "\n";
        }
    }
 
    // ── Function ─────────────────────────────────────────────────────────────
 
    void emit_function(const Function& fn)
    {
        // "fn" or "async fn"
        out_ << (fn.is_async ? "async fn " : "fn ");
        out_ << fn.name << "(";
 
        for (size_t i = 0; i < fn.params.size(); ++i)
        {
            auto& p = fn.params[i];
            out_ << p->name << ": " << type_str(p->type);
            if (i + 1 < fn.params.size()) out_ << ", ";
        }
 
        // Return type derived from fn type
        std::string ret = "void";
        if (fn.type && fn.type->kind == Type::Kind::Fn && fn.type->ret_type())
            ret = type_str(fn.type->ret_type());
        out_ << ") -> " << ret << ":\n";
 
        push();
        for (auto& bb : fn.blocks)
            emit_block(*bb);
        pop();
    }
 
    // ── Basic block ──────────────────────────────────────────────────────────
 
    void emit_block(const BasicBlock& bb)
    {
        // Block label at current indent (one level less than instructions)
        out_ << std::string(indent_ - 2 < 0 ? 0 : indent_ - 2, ' ')
             << bb.label << ":\n";
 
        for (auto& inst : bb.insts)
            emit_inst(*inst);
    }
 
    // ── Instruction dispatcher ────────────────────────────────────────────────
 
    void emit_inst(const Instruction& inst)
    {
        // Dispatch to the correct print_* helper via dynamic_cast chain.
        // Order matters for subclass disambiguation — more derived first.
 
        if (auto* i = dynamic_cast<const BinOpInst*>(&inst))      { print_binop(*i);       return; }
        if (auto* i = dynamic_cast<const UnOpInst*>(&inst))       { print_unop(*i);        return; }
        if (auto* i = dynamic_cast<const CmpInst*>(&inst))        { print_cmp(*i);         return; }
        if (auto* i = dynamic_cast<const AllocaInst*>(&inst))     { print_alloca(*i);      return; }
        if (auto* i = dynamic_cast<const LoadInst*>(&inst))       { print_load(*i);        return; }
        if (auto* i = dynamic_cast<const StoreInst*>(&inst))      { print_store(*i);       return; }
        if (auto* i = dynamic_cast<const CondBranchInst*>(&inst)) { print_condbr(*i);      return; }
        if (auto* i = dynamic_cast<const BranchInst*>(&inst))     { print_br(*i);          return; }
        if (auto* i = dynamic_cast<const ReturnInst*>(&inst))     { print_ret(*i);         return; }
        if (auto* i = dynamic_cast<const CallInst*>(&inst))       { print_call(*i);        return; }
        if (auto* i = dynamic_cast<const PhiInst*>(&inst))        { print_phi(*i);         return; }
        if (auto* i = dynamic_cast<const TensorOpInst*>(&inst))   { print_tensor_op(*i);   return; }
        if (auto* i = dynamic_cast<const SpawnInst*>(&inst))      { print_spawn(*i);       return; }
        if (auto* i = dynamic_cast<const AwaitInst*>(&inst))      { print_await(*i);       return; }
        if (auto* i = dynamic_cast<const ParallelForInst*>(&inst)){ print_parallel_for(*i);return; }
        if (auto* i = dynamic_cast<const ParallelMapInst*>(&inst)){ print_parallel_map(*i);return; }
        if (auto* i = dynamic_cast<const BarrierInst*>(&inst))    { print_barrier();       return; }
        if (auto* i = dynamic_cast<const CastInst*>(&inst))       { print_cast(*i);        return; }
        if (auto* i = dynamic_cast<const ReshapeInst*>(&inst))    { print_reshape(*i);     return; }
 
        // Fallback — should never happen if all instruction types are covered
        line("; <unknown instruction: " + inst.name + ">");
    }
 
    // ── Per-instruction printers ──────────────────────────────────────────────
 
    void print_binop(const BinOpInst& i)
    {
        line(i.name + " = " + binop_str(i.op)
             + " " + val(i.lhs) + ", " + val(i.rhs)
             + type_comment(i.type));
    }
 
    void print_unop(const UnOpInst& i)
    {
        line(i.name + " = " + unop_str(i.op)
             + " " + val(i.operand)
             + type_comment(i.type));
    }
 
    void print_cmp(const CmpInst& i)
    {
        line(i.name + " = cmp." + cmp_str(i.cmp)
             + " " + val(i.lhs) + ", " + val(i.rhs)
             + type_comment(i.type));
    }
 
    void print_alloca(const AllocaInst& i)
    {
        line(i.name + " = alloca " + type_str(i.alloc_type));
    }
 
    void print_load(const LoadInst& i)
    {
        line(i.name + " = load " + val(i.ptr)
             + type_comment(i.type));
    }
 
    void print_store(const StoreInst& i)
    {
        line("store " + val(i.val) + " -> " + val(i.ptr));
    }
 
    void print_br(const BranchInst& i)
    {
        line("br " + block_label(i.target));
    }
 
    void print_condbr(const CondBranchInst& i)
    {
        line("br " + val(i.cond)
             + ", " + block_label(i.true_bb)
             + ", " + block_label(i.false_bb));
    }
 
    void print_ret(const ReturnInst& i)
    {
        if (i.val)
            line("ret " + val(*i.val));
        else
            line("ret void");
    }
 
    void print_call(const CallInst& i)
    {
        std::string s;
        if (!i.name.empty()) s = i.name + " = ";
        if (i.is_tail) s += "tail ";
        s += "call ";
        if (auto* fn = dynamic_cast<ir::Function*>(i.callee.get())) {
            const std::string& fname = fn->name;
            // Only prepend @ if it's missing
            if (!fname.empty() && fname[0] == '@') {
                s += fname;
            } else {
                s += "@" + fname;
            }
        } else {
            s += val(i.callee);
        }
        s += "(";
        for (size_t a = 0; a < i.args.size(); ++a)
        {
            s += val(i.args[a]);
            if (a + 1 < i.args.size()) s += ", ";
        }
        s += ")" + type_comment(i.type);
        line(s);
    }
 
    void print_phi(const PhiInst& i)
    {
        std::string s = i.name + " = phi ";
        for (size_t k = 0; k < i.incoming.size(); ++k)
        {
            s += "[" + val(i.incoming[k].first)
               + ", %" + i.incoming[k].second->label + "]";
            if (k + 1 < i.incoming.size()) s += ", ";
        }
        s += type_comment(i.type);
        line(s);
    }
 
    void print_tensor_op(const TensorOpInst& i)
    {
        std::string s;
        if (!i.name.empty()) s = i.name + " = ";
        s += "tensor." + tensor_op_str(i.op) + " ";
        for (size_t a = 0; a < i.args.size(); ++a)
        {
            s += val(i.args[a]);
            if (a + 1 < i.args.size()) s += ", ";
        }
        // Annotate inferred shape if present
        if (i.inferred_shape)
        {
            s += "  {shape=[";
            for (size_t d = 0; d < i.inferred_shape->size(); ++d)
            {
                s += std::to_string((*i.inferred_shape)[d]);
                if (d + 1 < i.inferred_shape->size()) s += ",";
            }
            s += "]}";
        }
        if (i.requires_grad) s += " #grad";
        s += type_comment(i.type);
        line(s);
    }
 
    void print_spawn(const SpawnInst& i)
    {
        line(i.name + " = spawn " + val(i.task) + type_comment(i.type));
    }
 
    void print_await(const AwaitInst& i)
    {
        line(i.name + " = await " + val(i.handle) + type_comment(i.type));
    }
 
    void print_parallel_for(const ParallelForInst& i)
    {
        line("parallel_for " + val(i.n) + ", " + val(i.body_fn));
    }
 
    void print_parallel_map(const ParallelMapInst& i)
    {
        line(i.name + " = parallel_map " + val(i.array) + ", " + val(i.map_fn)
             + type_comment(i.type));
    }
 
    void print_barrier()
    {
        line("barrier");
    }
 
    void print_cast(const CastInst& i)
    {
        line(i.name + " = cast " + val(i.src)
             + " to " + type_str(i.target_type)
             + type_comment(i.type));
    }
 
    void print_reshape(const ReshapeInst& i)
    {
        line(i.name + " = reshape " + val(i.tensor)
             + ", " + val(i.shape)
             + type_comment(i.type));
    }
 
    // ── Constant value emitter (for globals) ─────────────────────────────────
 
    void emit_constant_value(const Value& v)
    {
        if (auto* c = dynamic_cast<const ConstantInt*>(&v))
        {
            out_ << c->value;
        }
        else if (auto* c = dynamic_cast<const ConstantFloat*>(&v))
        {
            out_ << c->value;
        }
        else if (auto* c = dynamic_cast<const ConstantBool*>(&v))
        {
            out_ << (c->value ? "true" : "false");
        }
        else if (auto* c = dynamic_cast<const ConstantString*>(&v))
        {
            out_ << "\"" << c->value << "\"";
        }
        else if (auto* c = dynamic_cast<const ConstantTensor*>(&v))
        {
            out_ << "tensor[";
            for (size_t d = 0; d < c->shape.size(); ++d)
            {
                out_ << c->shape[d];
                if (d + 1 < c->shape.size()) out_ << "x";
            }
            out_ << "]";
        }
        else
        {
            out_ << v.name;
        }
    }
 
    std::string val(const ValuePtr& v) const
    {
        if (!v) return "<null>";

        // Inline small constants instead of a separate definition
        if (auto* c = dynamic_cast<const ConstantInt*>(v.get()))
            return std::to_string(c->value);
        if (auto* c = dynamic_cast<const ConstantFloat*>(v.get()))
        {
            std::ostringstream ss;
            ss << c->value;
            return ss.str();
        }
        if (auto* c = dynamic_cast<const ConstantBool*>(v.get()))
            return c->value ? "true" : "false";
        if (auto* c = dynamic_cast<const ConstantString*>(v.get()))
            return "\"" + c->value + "\"";
        if (auto* fn = dynamic_cast<const ir::Function*>(v.get())) {
            if (!fn->name.empty() && fn->name[0] == '@') return fn->name;
            return "@" + fn->name;
        }
        // Named value (SSA register, function, argument)
        return v->name;
    }
 
    // ── Type printer ──────────────────────────────────────────────────────────
 
    static std::string type_str(const TypePtr& t)
    {
        if (!t) return "?";
        switch (t->kind)
        {
            case Type::Kind::Void:   return "void";
            case Type::Kind::Bool:   return "bool";
            case Type::Kind::I32:    return "i32";
            case Type::Kind::I64:    return "i64";
            case Type::Kind::F32:    return "f32";
            case Type::Kind::F64:    return "f64";
            case Type::Kind::Str:    return "str";
            case Type::Kind::Infer:  return "_";
            case Type::Kind::Array:
                return "Array<" + type_str(t->inner_type()) + ">";
            case Type::Kind::Tensor:
                return "Tensor<" + type_str(t->inner_type()) + ">";
            case Type::Kind::Fn:
            {
                std::string s = "fn(";
                const auto& params = t->param_types();
                for (size_t i = 0; i < params.size(); ++i)
                {
                    s += type_str(params[i]);
                    if (i + 1 < params.size()) s += ", ";
                }
                s += ") -> " + type_str(t->ret_type());
                return s;
            }
        }
        return "?";
    }
 
    std::string type_comment(const TypePtr& t) const
    {
        return "  ; " + type_str(t);
    }
 
    // ── Block label helper ────────────────────────────────────────────────────
 
    static std::string block_label(const BasicBlock* bb)
    {
        return bb ? ("%" + bb->label) : "%<null>";
    }
 
    // ── Opcode name tables ────────────────────────────────────────────────────
 
    static std::string binop_str(BinOpCode op)
    {
        switch (op)
        {
            case BinOpCode::Add:  return "add";
            case BinOpCode::Sub:  return "sub";
            case BinOpCode::Mul:  return "mul";
            case BinOpCode::Div:  return "div";
            case BinOpCode::Mod:  return "mod";
            case BinOpCode::And:  return "and";
            case BinOpCode::Or:   return "or";
            case BinOpCode::Xor:  return "xor";
            case BinOpCode::Shl:  return "shl";
            case BinOpCode::Shr:  return "shr";
            case BinOpCode::FAdd: return "fadd";
            case BinOpCode::FSub: return "fsub";
            case BinOpCode::FMul: return "fmul";
            case BinOpCode::FDiv: return "fdiv";
        }
        return "?";
    }
 
    static std::string unop_str(UnOpCode op)
    {
        switch (op)
        {
            case UnOpCode::Neg:  return "neg";
            case UnOpCode::Not:  return "not";
            case UnOpCode::FNeg: return "fneg";
        }
        return "?";
    }
 
    static std::string cmp_str(CmpCode c)
    {
        switch (c)
        {
            case CmpCode::Eq: return "eq";
            case CmpCode::Ne: return "ne";
            case CmpCode::Lt: return "lt";
            case CmpCode::Le: return "le";
            case CmpCode::Gt: return "gt";
            case CmpCode::Ge: return "ge";
        }
        return "?";
    }
 
    static std::string tensor_op_str(TensorOpCode op)
    {
        switch (op)
        {
            // Creation
            case TensorOpCode::Zeros:     return "zeros";
            case TensorOpCode::Ones:      return "ones";
            case TensorOpCode::Full:      return "full";
            case TensorOpCode::Eye:       return "eye";
            case TensorOpCode::Arange:    return "arange";
            case TensorOpCode::Linspace:  return "linspace";
            case TensorOpCode::Rand:      return "rand";
            case TensorOpCode::Randn:     return "randn";
            case TensorOpCode::RandInt:   return "randint";
            case TensorOpCode::FromList:  return "from_list";
            // Shape
            case TensorOpCode::Reshape:   return "reshape";
            case TensorOpCode::View:      return "view";
            case TensorOpCode::Flatten:   return "flatten";
            case TensorOpCode::Squeeze:   return "squeeze";
            case TensorOpCode::Unsqueeze: return "unsqueeze";
            case TensorOpCode::Transpose: return "transpose";
            case TensorOpCode::Permute:   return "permute";
            case TensorOpCode::Contiguous:return "contiguous";
            case TensorOpCode::Clone:     return "clone";
            case TensorOpCode::Cast:      return "cast";
            // Slice / join
            case TensorOpCode::Slice:     return "slice";
            case TensorOpCode::Select:    return "select";
            case TensorOpCode::Cat:       return "cat";
            case TensorOpCode::Stack:     return "stack";
            case TensorOpCode::Split:     return "split";
            case TensorOpCode::Chunk:     return "chunk";
            case TensorOpCode::Tile:      return "tile";
            case TensorOpCode::Repeat:    return "repeat";
            case TensorOpCode::Pad:       return "pad";
            // Reductions
            case TensorOpCode::Sum:       return "sum";
            case TensorOpCode::Mean:      return "mean";
            case TensorOpCode::Max:       return "max";
            case TensorOpCode::Min:       return "min";
            case TensorOpCode::Prod:      return "prod";
            case TensorOpCode::Norm:      return "norm";
            case TensorOpCode::Std:       return "std";
            case TensorOpCode::Var:       return "var";
            case TensorOpCode::Median:    return "median";
            case TensorOpCode::SumDim:    return "sum_dim";
            case TensorOpCode::MeanDim:   return "mean_dim";
            case TensorOpCode::MaxDim:    return "max_dim";
            case TensorOpCode::MinDim:    return "min_dim";
            case TensorOpCode::ArgMax:    return "argmax";
            case TensorOpCode::ArgMin:    return "argmin";
            case TensorOpCode::AllDim:    return "all_dim";
            case TensorOpCode::AnyDim:    return "any_dim";
            case TensorOpCode::CumSum:    return "cumsum";
            case TensorOpCode::CumProd:   return "cumprod";
            // Element-wise math
            case TensorOpCode::Exp:       return "exp";
            case TensorOpCode::Log:       return "log";
            case TensorOpCode::Log2:      return "log2";
            case TensorOpCode::Log1p:     return "log1p";
            case TensorOpCode::Sqrt:      return "sqrt";
            case TensorOpCode::Rsqrt:     return "rsqrt";
            case TensorOpCode::Abs:       return "abs";
            case TensorOpCode::Sign:      return "sign";
            case TensorOpCode::Sin:       return "sin";
            case TensorOpCode::Cos:       return "cos";
            case TensorOpCode::Tan:       return "tan";
            case TensorOpCode::Floor:     return "floor";
            case TensorOpCode::Ceil:      return "ceil";
            case TensorOpCode::Round:     return "round";
            case TensorOpCode::Neg:       return "neg";
            case TensorOpCode::Reciprocal:return "reciprocal";
            case TensorOpCode::Pow:       return "pow";
            case TensorOpCode::Clamp:     return "clamp";
            case TensorOpCode::Lerp:      return "lerp";
            // Activations
            case TensorOpCode::Relu:      return "relu";
            case TensorOpCode::Relu6:     return "relu6";
            case TensorOpCode::Silu:      return "silu";
            case TensorOpCode::Gelu:      return "gelu";
            case TensorOpCode::Sigmoid:   return "sigmoid";
            case TensorOpCode::Tanh:      return "tanh";
            case TensorOpCode::Softmax:   return "softmax";
            case TensorOpCode::LogSoftmax:return "log_softmax";
            case TensorOpCode::Hardsigmoid:return "hardsigmoid";
            case TensorOpCode::Hardswish: return "hardswish";
            case TensorOpCode::Mish:      return "mish";
            case TensorOpCode::LeakyRelu: return "leaky_relu";
            case TensorOpCode::Elu:       return "elu";
            case TensorOpCode::Celu:      return "celu";
            case TensorOpCode::Selu:      return "selu";
            case TensorOpCode::Prelu:     return "prelu";
            // Linear algebra
            case TensorOpCode::Dot:       return "dot";
            case TensorOpCode::MatMul:    return "matmul";
            case TensorOpCode::Bmm:       return "bmm";
            case TensorOpCode::Outer:     return "outer";
            case TensorOpCode::Cross:     return "cross";
            case TensorOpCode::Kron:      return "kron";
            case TensorOpCode::Inverse:   return "inverse";
            case TensorOpCode::PInverse:  return "pinverse";
            case TensorOpCode::Det:       return "det";
            case TensorOpCode::Trace:     return "trace";
            case TensorOpCode::Diag:      return "diag";
            case TensorOpCode::Triu:      return "triu";
            case TensorOpCode::Tril:      return "tril";
            case TensorOpCode::Svd:       return "svd";
            case TensorOpCode::Eig:       return "eig";
            case TensorOpCode::Qr:        return "qr";
            case TensorOpCode::Cholesky:  return "cholesky";
            case TensorOpCode::Solve:     return "solve";
            // Sort / index
            case TensorOpCode::Sort:      return "sort";
            case TensorOpCode::ArgSort:   return "argsort";
            case TensorOpCode::TopK:      return "topk";
            case TensorOpCode::Gather:    return "gather";
            case TensorOpCode::Scatter:   return "scatter";
            case TensorOpCode::Where:     return "where";
            case TensorOpCode::NonZero:   return "nonzero";
            case TensorOpCode::MaskedSelect: return "masked_select";
            // Autodiff
            case TensorOpCode::Backward:  return "backward";
            case TensorOpCode::Grad:      return "grad";
            case TensorOpCode::NoGrad:    return "no_grad";
            case TensorOpCode::Detach:    return "detach";
            case TensorOpCode::ZeroGrad:  return "zero_grad";
            case TensorOpCode::RequiresGrad: return "requires_grad";
        }
        return "?";
    }
};
 
} // namespace ir