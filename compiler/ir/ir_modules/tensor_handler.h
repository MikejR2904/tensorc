#pragma once

#include "../../io/module_handler.h"
#include "../IRBuilder.h"
#include "../Instruction.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace ir {

/// Handler for the built-in "tensor" module.
/// Lowers tensor operations to TensorOpInst IR instructions.
class TensorModuleHandler : public io::ModuleHandler {
public:
    ir::Value* lower_call(ir::IRBuilder* builder, const std::string& func_name, const std::vector<ir::ValuePtr>& args, const TypePtr& ret_type) override 
    {
        // Attempt to resolve the function name to a tensor operation
        TensorOpCode op = resolve_op(func_name);
        if (op == TensorOpCode::Unknown) {
            return nullptr;  // Cannot handle this function
        }

        // Resolve result type safely: prefer sema annotation, fall back to first
        // well-typed tensor arg's type, then default to Tensor<f32>.
        // Guards against null elem_type() which causes a crash.
        TypePtr effective_ty = ret_type;
        bool needs_resolve = !effective_ty || effective_ty->is_infer() ||
            (effective_ty->kind == Type::Kind::Tensor && 
             (!effective_ty->elem_type() || effective_ty->elem_type()->is_infer()));

        if (needs_resolve) {
            TypePtr elem_ty = Type::f32();
            for (const auto& arg : args) {
                if (!arg || !arg->type) continue;
                if (arg->type->kind == Type::Kind::Tensor) {
                    TypePtr el = arg->type->elem_type();
                    if (el && !el->is_infer()) { 
                        elem_ty = el; 
                        break; 
                    }
                }
            }
            effective_ty = Type::tensor(elem_ty);
        }

        // For ops known to return scalar (reduce-to-number)
        if (op == TensorOpCode::Sum || op == TensorOpCode::Mean || op == TensorOpCode::Max || op == TensorOpCode::Min || op == TensorOpCode::Prod) {
            effective_ty = Type::f32();
        }

        // Emit the tensor operation instruction
        return builder->emit<TensorOpInst>(op_is_void(op) ? "" : builder->fresh(), effective_ty, op, args);
    }

    std::string module_name() const override { return "tensor"; }
    bool is_builtin() const override { return true; }
    bool supports_gpu() const override { return true; }

private:
    /// Map function name to TensorOpCode.
    /// Returns TensorOpCode::Unknown if not recognized.
    TensorOpCode resolve_op(const std::string& name) const {
        static const std::unordered_map<std::string, TensorOpCode> ops = {
            {"zeros", TensorOpCode::Zeros},
            {"ones", TensorOpCode::Ones},
            {"full", TensorOpCode::Full},
            {"eye", TensorOpCode::Eye},
            {"arange", TensorOpCode::Arange},
            {"linspace", TensorOpCode::Linspace},
            {"rand", TensorOpCode::Rand},
            {"randn", TensorOpCode::Randn},
            {"randint", TensorOpCode::RandInt},
            {"from_list", TensorOpCode::FromList},
            {"reshape", TensorOpCode::Reshape},
            {"view", TensorOpCode::View},
            {"flatten", TensorOpCode::Flatten},
            {"squeeze", TensorOpCode::Squeeze},
            {"unsqueeze", TensorOpCode::Unsqueeze},
            {"transpose", TensorOpCode::Transpose},
            {"permute", TensorOpCode::Permute},
            {"contiguous", TensorOpCode::Contiguous},
            {"clone", TensorOpCode::Clone},
            {"slice", TensorOpCode::Slice},
            {"select", TensorOpCode::Select},
            {"cat", TensorOpCode::Cat},
            {"stack", TensorOpCode::Stack},
            {"split", TensorOpCode::Split},
            {"chunk", TensorOpCode::Chunk},
            {"tile", TensorOpCode::Tile},
            {"repeat", TensorOpCode::Repeat},
            {"pad", TensorOpCode::Pad},
            {"sum", TensorOpCode::Sum},
            {"mean", TensorOpCode::Mean},
            {"max", TensorOpCode::Max},
            {"min", TensorOpCode::Min},
            {"prod", TensorOpCode::Prod},
            {"norm", TensorOpCode::Norm},
            {"std", TensorOpCode::Std},
            {"var", TensorOpCode::Var},
            {"median", TensorOpCode::Median},
            {"argmax", TensorOpCode::ArgMax},
            {"argmin", TensorOpCode::ArgMin},
            {"exp", TensorOpCode::Exp},
            {"log", TensorOpCode::Log},
            {"log2", TensorOpCode::Log2},
            {"log1p", TensorOpCode::Log1p},
            {"sqrt", TensorOpCode::Sqrt},
            {"rsqrt", TensorOpCode::Rsqrt},
            {"abs", TensorOpCode::Abs},
            {"sign", TensorOpCode::Sign},
            {"sin", TensorOpCode::Sin},
            {"cos", TensorOpCode::Cos},
            {"tan", TensorOpCode::Tan},
            {"floor", TensorOpCode::Floor},
            {"ceil", TensorOpCode::Ceil},
            {"round", TensorOpCode::Round},
            {"pow", TensorOpCode::Pow},
            {"clamp", TensorOpCode::Clamp},
            {"lerp", TensorOpCode::Lerp},
            {"relu", TensorOpCode::Relu},
            {"relu6", TensorOpCode::Relu6},
            {"silu", TensorOpCode::Silu},
            {"gelu", TensorOpCode::Gelu},
            {"sigmoid", TensorOpCode::Sigmoid},
            {"tanh", TensorOpCode::Tanh},
            {"softmax", TensorOpCode::Softmax},
            {"log_softmax", TensorOpCode::LogSoftmax},
            {"leaky_relu", TensorOpCode::LeakyRelu},
            {"elu", TensorOpCode::Elu},
            {"selu", TensorOpCode::Selu},
            {"prelu", TensorOpCode::Prelu},
            {"dot", TensorOpCode::Dot},
            {"matmul", TensorOpCode::MatMul},
            {"bmm", TensorOpCode::Bmm},
            {"outer", TensorOpCode::Outer},
            {"cross", TensorOpCode::Cross},
            {"inverse", TensorOpCode::Inverse},
            {"det", TensorOpCode::Det},
            {"trace", TensorOpCode::Trace},
            {"svd", TensorOpCode::Svd},
            {"eig", TensorOpCode::Eig},
            {"qr", TensorOpCode::Qr},
            {"cholesky", TensorOpCode::Cholesky},
            {"solve", TensorOpCode::Solve},
            {"sort", TensorOpCode::Sort},
            {"argsort", TensorOpCode::ArgSort},
            {"topk", TensorOpCode::TopK},
            {"gather", TensorOpCode::Gather},
            {"scatter", TensorOpCode::Scatter},
            {"where", TensorOpCode::Where},
            {"nonzero", TensorOpCode::NonZero},
            {"backward", TensorOpCode::Backward},
            {"grad", TensorOpCode::Grad},
            {"no_grad", TensorOpCode::NoGrad},
            {"detach", TensorOpCode::Detach},
            {"zero_grad", TensorOpCode::ZeroGrad},
            {"requires_grad", TensorOpCode::RequiresGrad},
        };

        auto it = ops.find(name);
        return (it != ops.end()) ? it->second : TensorOpCode::Unknown;
    }
    
    /// Check if an operation returns void (no result value).
    bool op_is_void(TensorOpCode op) const {
        return op == TensorOpCode::Backward || op == TensorOpCode::ZeroGrad || op == TensorOpCode::NoGrad;
    }
};

} // namespace ir