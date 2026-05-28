#include "InstrSelector.h"
#include "MachineInstr.h"

/// codegen/InstrSelector.cpp
///
/// Lowers IR instructions to RISC-V 64-bit (RV64GV) MachineInstr sequences.
///
/// Virtual register IDs
/// ────────────────────
/// Every IR Value* is mapped to a stable small integer via mf->vreg_for(ptr).
/// This eliminates the reinterpret_cast<intptr_t> pointer-truncation bug that
/// caused silent vreg collisions on 64-bit builds (two different Values could
/// map to the same 16-bit truncated pointer value).
///
/// Tensor operations
/// ─────────────────
/// Scalar:  BinOpInst → add / sub / mul / div (RV64I M-extension)
/// Vector:  ElemAdd/ElemMul → vsetvli + vadd.vv/vmul.vv (RVV 1.0)
/// MatMul:  MatMul/Bmm → call matmul_tile_f64 (tiled kernel)
/// Fused:   FusedMatMulRelu/Gelu/Silu/Tanh → call matmul_<act>_tile_f64
///
/// Control flow
/// ────────────
/// CondBranchInst → blt/bge (mapped from CmpCode in a preceding CmpInst).
/// BranchInst     → j (unconditional jump).
/// ReturnInst     → mv a0, <vreg>  (if returning a value) + ret.

namespace codegen {

// ── Scalar / arithmetic ──────────────────────────────────────────────────────

void InstrSelector::visit(ir::BinOpInst& inst)
{
    int dst = vreg(&inst);
    int lhs = vreg(inst.lhs.get());
    int rhs = vreg(inst.rhs.get());

    const char* opc = "add";
    bool fp = inst.type && (inst.type->kind == Type::Kind::F32 ||
                            inst.type->kind == Type::Kind::F64);
    switch (inst.op) {
    case ir::BinOpCode::Add:  opc = fp ? "fadd.d" : "add";  break;
    case ir::BinOpCode::Sub:  opc = fp ? "fsub.d" : "sub";  break;
    case ir::BinOpCode::Mul:  opc = fp ? "fmul.d" : "mul";  break;
    case ir::BinOpCode::Div:  opc = fp ? "fdiv.d" : "div";  break;
    case ir::BinOpCode::FAdd: opc = "fadd.d"; break;
    case ir::BinOpCode::FSub: opc = "fsub.d"; break;
    case ir::BinOpCode::FMul: opc = "fmul.d"; break;
    case ir::BinOpCode::FDiv: opc = "fdiv.d"; break;
    case ir::BinOpCode::And:  opc = "and";  break;
    case ir::BinOpCode::Or:   opc = "or";   break;
    case ir::BinOpCode::Xor:  opc = "xor";  break;
    case ir::BinOpCode::Shl:  opc = "sll";  break;
    case ir::BinOpCode::Shr:  opc = "srl";  break;
    default: opc = "add"; break;
    }

    emit(MachineInstr::make(opc)
        .add(MachineOperand::vreg(dst))
        .add(MachineOperand::vreg(lhs))
        .add(MachineOperand::vreg(rhs)));
}

void InstrSelector::visit(ir::UnOpInst& inst)
{
    int dst = vreg(&inst);
    int src = vreg(inst.operand.get());
    switch (inst.op) {
    case ir::UnOpCode::Neg:
        // neg rd, rs  = sub rd, x0, rs
        emit(MachineInstr::make("sub")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::preg(0))      // x0
            .add(MachineOperand::vreg(src)));
        break;
    case ir::UnOpCode::FNeg:
        emit(MachineInstr::make("fneg.d")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::vreg(src)));
        break;
    case ir::UnOpCode::Not:
        // not rd, rs  = xori rd, rs, -1
        emit(MachineInstr::make("xori")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::vreg(src))
            .add(MachineOperand::imm_op(-1)));
        break;
    }
}

void InstrSelector::visit(ir::CmpInst& inst)
{
    // CmpInst produces a boolean (0/1) result used by CondBranchInst.
    // We emit a slt/sltu sequence; the branch visitor picks the branch opcode.
    int dst = vreg(&inst);
    int lhs = vreg(inst.lhs.get());
    int rhs = vreg(inst.rhs.get());

    switch (inst.cmp) {
    case ir::CmpCode::Lt:
        emit(MachineInstr::make("slt")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::vreg(lhs))
            .add(MachineOperand::vreg(rhs)));
        break;
    case ir::CmpCode::Le:
        // le: not (rhs < lhs)  →  slt tmp, rhs, lhs; xori dst, tmp, 1
        {
            int tmp = mf->vregs.next();
            emit(MachineInstr::make("slt")
                .add(MachineOperand::vreg(tmp))
                .add(MachineOperand::vreg(rhs))
                .add(MachineOperand::vreg(lhs)));
            emit(MachineInstr::make("xori")
                .add(MachineOperand::vreg(dst))
                .add(MachineOperand::vreg(tmp))
                .add(MachineOperand::imm_op(1)));
        }
        break;
    case ir::CmpCode::Gt:
        emit(MachineInstr::make("slt")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::vreg(rhs))
            .add(MachineOperand::vreg(lhs)));
        break;
    case ir::CmpCode::Ge:
        {
            int tmp = mf->vregs.next();
            emit(MachineInstr::make("slt")
                .add(MachineOperand::vreg(tmp))
                .add(MachineOperand::vreg(lhs))
                .add(MachineOperand::vreg(rhs)));
            emit(MachineInstr::make("xori")
                .add(MachineOperand::vreg(dst))
                .add(MachineOperand::vreg(tmp))
                .add(MachineOperand::imm_op(1)));
        }
        break;
    case ir::CmpCode::Eq:
        {
            // sub tmp, lhs, rhs; seqz dst, tmp
            int tmp = mf->vregs.next();
            emit(MachineInstr::make("sub")
                .add(MachineOperand::vreg(tmp))
                .add(MachineOperand::vreg(lhs))
                .add(MachineOperand::vreg(rhs)));
            emit(MachineInstr::make("seqz")
                .add(MachineOperand::vreg(dst))
                .add(MachineOperand::vreg(tmp)));
        }
        break;
    case ir::CmpCode::Ne:
        {
            int tmp = mf->vregs.next();
            emit(MachineInstr::make("sub")
                .add(MachineOperand::vreg(tmp))
                .add(MachineOperand::vreg(lhs))
                .add(MachineOperand::vreg(rhs)));
            emit(MachineInstr::make("snez")
                .add(MachineOperand::vreg(dst))
                .add(MachineOperand::vreg(tmp)));
        }
        break;
    }
}

// ── Memory ───────────────────────────────────────────────────────────────────

void InstrSelector::visit(ir::LoadInst& inst)
{
    int dst = vreg(&inst);
    int ptr = vreg(inst.ptr.get());
    emit(MachineInstr::make("ld")
        .add(MachineOperand::vreg(dst))
        .add(MachineOperand::mem(ptr, 0, true)));
}

void InstrSelector::visit(ir::StoreInst& inst)
{
    int val = vreg(inst.val.get());
    int ptr = vreg(inst.ptr.get());
    emit(MachineInstr::make("sd")
        .add(MachineOperand::vreg(val))
        .add(MachineOperand::mem(ptr, 0, true)));
}

// ── Tensor operations ─────────────────────────────────────────────────────────

void InstrSelector::visit(ir::TensorOpInst& inst)
{
    int dst = vreg(&inst);
    auto arg_reg = [&](size_t i) {
        return (i < inst.args.size()) ? vreg(inst.args[i].get()) : 0;
    };

    switch (inst.op) {

    // ── Element-wise (RVV) ────────────────────────────────────────────────
    case ir::TensorOpCode::ElemAdd: {
        // vsetvli t0, a0, e64, m1, ta, ma
        // vadd.vv vd, vs1, vs2
        emit(MachineInstr::make("vsetvli")
            .add(MachineOperand::preg(5))           // t0
            .add(MachineOperand::preg(10))          // a0 (element count)
            .add(MachineOperand::imm_op(0x03)));    // e64, m1 encoding
        emit(MachineInstr::make("vadd.vv")
            .add(MachineOperand::vreg(dst, RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(0), RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(1), RegClass::Vector)));
        break;
    }
    case ir::TensorOpCode::ElemSub: {
        emit(MachineInstr::make("vsetvli")
            .add(MachineOperand::preg(5))
            .add(MachineOperand::preg(10))
            .add(MachineOperand::imm_op(0x03)));
        emit(MachineInstr::make("vsub.vv")
            .add(MachineOperand::vreg(dst, RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(0), RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(1), RegClass::Vector)));
        break;
    }
    case ir::TensorOpCode::ElemMul: {
        emit(MachineInstr::make("vsetvli")
            .add(MachineOperand::preg(5))
            .add(MachineOperand::preg(10))
            .add(MachineOperand::imm_op(0x03)));
        emit(MachineInstr::make("vmul.vv")
            .add(MachineOperand::vreg(dst, RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(0), RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(1), RegClass::Vector)));
        break;
    }
    case ir::TensorOpCode::ElemDiv: {
        emit(MachineInstr::make("vsetvli")
            .add(MachineOperand::preg(5))
            .add(MachineOperand::preg(10))
            .add(MachineOperand::imm_op(0x03)));
        emit(MachineInstr::make("vdiv.vv")
            .add(MachineOperand::vreg(dst, RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(0), RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(1), RegClass::Vector)));
        break;
    }

    // ── Activations (element-wise, scalar zero) ───────────────────────────
    case ir::TensorOpCode::Relu: {
        // vfmax.vf vd, vs, f0  (f0 holds 0.0)
        emit(MachineInstr::make("vsetvli")
            .add(MachineOperand::preg(5))
            .add(MachineOperand::preg(10))
            .add(MachineOperand::imm_op(0x03)));
        emit(MachineInstr::make("vfmax.vf")
            .add(MachineOperand::vreg(dst, RegClass::Vector))
            .add(MachineOperand::vreg(arg_reg(0), RegClass::Vector))
            .add(MachineOperand::preg(0, RegClass::FPR)));   // f0 = 0.0
        break;
    }
    case ir::TensorOpCode::Exp: {
        // No direct RVV exp — call vectorized libm kernel
        emit(MachineInstr::make("mv")
            .add(MachineOperand::preg(10))
            .add(MachineOperand::vreg(arg_reg(0))));
        emit(MachineInstr::make("call")
            .add(MachineOperand::label_op("vexp_f64")));
        emit(MachineInstr::make("mv")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::preg(10)));
        break;
    }

    // ── Fused elem-chain ──────────────────────────────────────────────────
    case ir::TensorOpCode::FusedElemChain: {
        // Emit a call to a runtime fused-kernel dispatcher with the chain
        // body recorded in inst.elem_body.  Each node becomes a scalar op
        // inside a single vector loop.  For now emit a pseudo-op that the
        // assembler/linker resolves to the generated kernel.
        emit(MachineInstr::make("# fused.elem_chain: " +
                                std::to_string(inst.elem_body.size()) + " nodes"));
        for (size_t k = 0; k < inst.args.size(); ++k) {
            emit(MachineInstr::make("mv")
                .add(MachineOperand::preg(10 + (int)k))
                .add(MachineOperand::vreg(arg_reg(k))));
        }
        emit(MachineInstr::make("call")
            .add(MachineOperand::label_op("fused_elem_chain_kernel")));
        emit(MachineInstr::make("mv")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::preg(10)));
        break;
    }

    // ── Matrix multiply ───────────────────────────────────────────────────
    case ir::TensorOpCode::MatMul:
    case ir::TensorOpCode::Bmm:
    case ir::TensorOpCode::Dot: {
        // Move args into a0, a1 and call the tiled matmul kernel
        emit(MachineInstr::make("mv")
            .add(MachineOperand::preg(10))
            .add(MachineOperand::vreg(arg_reg(0))));
        emit(MachineInstr::make("mv")
            .add(MachineOperand::preg(11))
            .add(MachineOperand::vreg(arg_reg(1))));
        emit(MachineInstr::make("call")
            .add(MachineOperand::label_op("matmul_tile_f64")));
        // Capture return value from a0
        emit(MachineInstr::make("mv")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::preg(10)));
        break;
    }

    // ── Fused MatMul + activation ─────────────────────────────────────────
    case ir::TensorOpCode::FusedMatMulRelu: {
        emit(MachineInstr::make("mv").add(MachineOperand::preg(10)).add(MachineOperand::vreg(arg_reg(0))));
        emit(MachineInstr::make("mv").add(MachineOperand::preg(11)).add(MachineOperand::vreg(arg_reg(1))));
        emit(MachineInstr::make("call").add(MachineOperand::label_op("matmul_relu_tile_f64")));
        emit(MachineInstr::make("mv").add(MachineOperand::vreg(dst)).add(MachineOperand::preg(10)));
        break;
    }
    case ir::TensorOpCode::FusedMatMulGelu: {
        emit(MachineInstr::make("mv").add(MachineOperand::preg(10)).add(MachineOperand::vreg(arg_reg(0))));
        emit(MachineInstr::make("mv").add(MachineOperand::preg(11)).add(MachineOperand::vreg(arg_reg(1))));
        emit(MachineInstr::make("call").add(MachineOperand::label_op("matmul_gelu_tile_f64")));
        emit(MachineInstr::make("mv").add(MachineOperand::vreg(dst)).add(MachineOperand::preg(10)));
        break;
    }
    case ir::TensorOpCode::FusedMatMulSilu: {
        emit(MachineInstr::make("mv").add(MachineOperand::preg(10)).add(MachineOperand::vreg(arg_reg(0))));
        emit(MachineInstr::make("mv").add(MachineOperand::preg(11)).add(MachineOperand::vreg(arg_reg(1))));
        emit(MachineInstr::make("call").add(MachineOperand::label_op("matmul_silu_tile_f64")));
        emit(MachineInstr::make("mv").add(MachineOperand::vreg(dst)).add(MachineOperand::preg(10)));
        break;
    }
    case ir::TensorOpCode::FusedMatMulTanh: {
        emit(MachineInstr::make("mv").add(MachineOperand::preg(10)).add(MachineOperand::vreg(arg_reg(0))));
        emit(MachineInstr::make("mv").add(MachineOperand::preg(11)).add(MachineOperand::vreg(arg_reg(1))));
        emit(MachineInstr::make("call").add(MachineOperand::label_op("matmul_tanh_tile_f64")));
        emit(MachineInstr::make("mv").add(MachineOperand::vreg(dst)).add(MachineOperand::preg(10)));
        break;
    }

    default: {
        // Unhandled tensor op: emit a labelled pseudo-call
        std::string tag = "tensor.op." + std::to_string(static_cast<int>(inst.op));
        emit(MachineInstr::make(tag).add(MachineOperand::vreg(dst)));
        break;
    }
    }
}

// ── Calls ─────────────────────────────────────────────────────────────────────

void InstrSelector::visit(ir::CallInst& inst)
{
    // ABI: first 8 args go in a0-a7 (x10-x17); rest are caller-saved stack slots.
    for (size_t i = 0; i < inst.args.size() && i < 8; ++i) {
        emit(MachineInstr::make("mv")
            .add(MachineOperand::preg(10 + (int)i))
            .add(MachineOperand::vreg(vreg(inst.args[i].get()))));
    }
    std::string callee = inst.callee ? inst.callee->name : "";
    if (!callee.empty() && callee[0] == '@') callee.erase(callee.begin());
    if (callee.empty()) callee = "unknown_callee";
    emit(MachineInstr::make("call")
        .add(MachineOperand::label_op(callee)));

    // Capture result from a0 if the call produces a value
    if (!inst.name.empty()) {
        int dst = vreg(&inst);
        emit(MachineInstr::make("mv")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::preg(10)));
    }
}

// ── Control flow ─────────────────────────────────────────────────────────────

void InstrSelector::visit(ir::ReturnInst& inst)
{
    // RISC-V ABI: return value in a0 (x10) for GPR, fa0 (f10) for FPR.
    if (inst.val) {
        int val_vreg = vreg(inst.val->get());
        // Explicit mv into a0 so RegAlloc sees this as the canonical return-value use.
        emit(MachineInstr::make("mv")
            .add(MachineOperand::preg(10))    // a0 — physical destination
            .add(MachineOperand::vreg(val_vreg)));
    }
    emit(MachineInstr::make("ret"));
}

void InstrSelector::visit(ir::PhiInst& inst)
{
    // Lower Phi to COPY pseudo-instructions between predecessor blocks.
    // Full SSA destruction requires inserting copies at block boundaries —
    // this prototype emits them in the current block as a conservative approximation.
    int dst = vreg(&inst);
    for (auto& [val_ptr, _bb] : inst.incoming) {
        int src = vreg(val_ptr.get());
        if (src == dst) continue; // trivial phi — skip
        auto mi = MachineInstr::make("mv")
            .add(MachineOperand::vreg(dst))
            .add(MachineOperand::vreg(src));
        mi.is_pseudo = true;
        emit(mi);
    }
}

void InstrSelector::visit(ir::BranchInst& inst)
{
    emit(MachineInstr::make("j")
        .add(MachineOperand::label_op(mf->label_for(inst.target))));
}

void InstrSelector::visit(ir::CondBranchInst& inst)
{
    int cond = vreg(inst.cond.get());
    // bnez cond, true_label; j false_label
    emit(MachineInstr::make("bnez")
        .add(MachineOperand::vreg(cond))
        .add(MachineOperand::label_op(mf->label_for(inst.true_bb))));
    emit(MachineInstr::make("j")
        .add(MachineOperand::label_op(mf->label_for(inst.false_bb))));
}

} // namespace codegen
