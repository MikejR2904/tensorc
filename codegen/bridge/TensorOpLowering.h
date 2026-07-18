#pragma once
 
/// codegen/bridge/TensorOpLowering.h
///
/// TensorOpLowering — The Bridge Between IR Middle-End and Codegen
///
/// It operates as the final IR pass, invoked after FusionPass and
/// TypePropPass have stabilised the IR.  For every TensorOpInst it:
///
///   Phase A  — Tiler:               TensorOpInst → LoopNest (LLIR)
///   Phase B1 — ScratchpadAllocator: LoopNest → abstract byte offsets
///   Phase B2 — StaticMMU:           abstract offsets → hardware bank addresses
///   Phase B3 — KReductionHandler:   inject K>64 flush/swap/writeback
///   Phase B4 — MemoryLegalizer:     finalise DMA ops
///   Phase C  — DependencyScheduler: producer-consumer reordering
///   Phase C' — Scheduler:           ping-pong double-buffering
///   Phase D  — RiscVTargetEmitter:  → Custom-0 .insn assembly text
///
/// The emitted assembly text is stored in TensorOpInst::lowered_asm.
/// The legacy AsmPrinter checks this field and, when non-empty, emits
/// the pre-lowered text verbatim instead of calling the legacy path.
/// The two pipelines therefore co-exist without conflict.
///
/// Integration into the compiler driver
/// ─────────────────────────────────────
///   // After IR construction and middle-end passes:
///   codegen::bridge::TensorOpLoweringPass lowering_pass("riscv64");
///   lowering_pass.run(*ir_module);
///   // Then call legacy codegen as normal — it will skip lowered ops.
///
/// See codegen/bridge/TensorOpLowering.cpp for implementation.
 
#include "../../compiler/ir/IRModule.h"
#include "../../compiler/ir/IRPasses.h"
#include "../../compiler/ir/Instruction.h"
#include "../lowering/Tiler.h"
#include "../lowering/ScratchpadAllocator.h"
#include "../lowering/MemoryLegalizer.h"
#include "../lowering/Scheduler.h"
#include "../targets/RiscVTargetEmitter.h"
#include "../targets/X86TargetEmitter.h"
#include <string>
#include <map>
#include <sstream>
#include <cstddef>
 
namespace codegen::bridge {
 
// ── ShapeInference ────────────────────────────────────────────────────────────
// Extracts concrete tensor shapes from a TensorOpInst's operand types.
// Returns an empty map when shape information is unavailable (caller falls
// back to the legacy scalar emission path).
 
struct ShapeInference {
    using ShapeMap = std::map<const void*, lowering::TensorShape>;
 
    static lowering::TensorShape shape_from_type(const TypePtr& t) {
        if (!t || t->kind != Type::Kind::Tensor) return {};
        lowering::TensorShape s;
        s.element_bytes = (t->elem_type() && t->elem_type()->kind == Type::Kind::F32) ? 4 : 8;
        for (const auto& dim : t->shape) {
            if (std::holds_alternative<int>(dim))
                s.dims.push_back(static_cast<int64_t>(std::get<int>(dim)));
            else
                s.dims.push_back(8);   // symbolic → default to tile size
        }
        if (s.dims.empty()) s.dims = {8, 8};
        return s;
    }
 
    static ShapeMap build(const ir::TensorOpInst& inst) {
        ShapeMap m;
        for (const auto& arg : inst.args) {
            if (!arg) continue;
            auto s = shape_from_type(arg->type);
            if (!s.dims.empty()) m[arg.get()] = s;
        }
        // Include the result shape
        auto rs = shape_from_type(inst.type);
        if (!rs.dims.empty()) m[&inst] = rs;
        return m;
    }
};
 
// ── TensorOpLoweringPass ──────────────────────────────────────────────────────
 
class TensorOpLoweringPass final : public ir::IPass {
public:
    explicit TensorOpLoweringPass(std::string target = "riscv64") : target_(std::move(target)) {}
 
    bool run(ir::IRModule& mod) override;
 
    std::string_view name() const override { return "TensorOpLoweringPass"; }
 
    /// Toggle verbose diagnostics for all sub-passes
    void set_verbose(bool v) { verbose_ = v; }
 
private:
    using ShapeMap = ShapeInference::ShapeMap;
    
    std::string target_;
    bool verbose_ = false;
    size_t lowered_count_ = 0;
 
    // ── Main dispatcher ──────────────────────────────────────────────────────
    /// Lower a single TensorOpInst through all pipeline phases.
    /// Returns the emitted assembly text, or empty string on failure
    /// (which causes the instruction to be left for legacy emission).
    std::string lower_one(ir::TensorOpInst& inst);
    
    // ── Opcode-specific handlers (all return assembly string or empty) ──────
    std::string lower_matmul(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_elemwise(ir::TensorOpInst& inst, const ShapeMap& shapes, const std::string& op);
    std::string lower_elemwise_math(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_activation(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_fused_matmul_activation(ir::TensorOpInst& inst, const ShapeMap& shapes, const std::string& activation);
    std::string lower_fused_elemwise_chain(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_reduction_full(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_reduction_dim(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_shape_op(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_creation(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_slice_join(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_linalg(ir::TensorOpInst& inst, const ShapeMap& shapes);
    std::string lower_sort_gather(ir::TensorOpInst& inst, const ShapeMap& shapes);
};
 
} // namespace codegen::bridge
