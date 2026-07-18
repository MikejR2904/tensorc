#include "RiscVInstrInfo.h"
#include "RiscVRegisterInfo.h"
#include <cstring>
#include <unordered_map>

namespace codegen::target::riscv64 {

namespace {
bool is_f32(LLT t) { return t == LLT::F32; }

MOperand zero() { return MOperand::Phys(X0, RegClass::GPR, LLT::I64); }

/// RISC-V syntax is destination-first: `op rd, rs1, rs2`.
MachineInstr rrr(unsigned op, int dst, MOperand a, MOperand b) {
    return MachineInstr::makeTarget(op).def(MOperand::Reg_(dst, LLT::I64)).use(a).use(b)
        .pr(MOperand::Reg_(dst, LLT::I64)).pr(a).pr(b);
}
} // namespace

const MCInstrDesc& RiscVInstrInfo::describe(unsigned target_op) const {
    static const std::unordered_map<unsigned, MCInstrDesc> table = {
        {MOV_RR,     {"mv",    true, false, false, false, false, false}},
        {LI,         {"li",    true, false, false, false, false, false}},
        {LEA_FRAME,  {"addi",  true, false, false, false, false, false}},
        {LOAD_D,     {"ld",    true, true,  false, false, false, false}},
        {LOAD_W,     {"lw",    true, true,  false, false, false, false}},
        {LOAD_HU,    {"lhu",   true, true,  false, false, false, false}},
        {LOAD_BU,    {"lbu",   true, true,  false, false, false, false}},
        {STORE_D,    {"sd",    false, false, true, false, false, false}},
        {STORE_W,    {"sw",    false, false, true, false, false, false}},
        {STORE_H,    {"sh",    false, false, true, false, false, false}},
        {STORE_B,    {"sb",    false, false, true, false, false, false}},
        {ADD_RRR,    {"add",   true, false, false, false, false, false}},
        {SUB_RRR,    {"sub",   true, false, false, false, false, false}},
        {AND_RRR,    {"and",   true, false, false, false, false, false}},
        {OR_RRR,     {"or",    true, false, false, false, false, false}},
        {XOR_RRR,    {"xor",   true, false, false, false, false, false}},
        {MUL_RRR,    {"mul",   true, false, false, false, false, false}},
        {SLL_RRR,    {"sll",   true, false, false, false, false, false}},
        {SRL_RRR,    {"srl",   true, false, false, false, false, false}},
        {SRA_RRR,    {"sra",   true, false, false, false, false, false}},
        {DIV_RRR,    {"div",   true, false, false, false, false, false}},
        {DIVU_RRR,   {"divu",  true, false, false, false, false, false}},
        {REM_RRR,    {"rem",   true, false, false, false, false, false}},
        {REMU_RRR,   {"remu",  true, false, false, false, false, false}},
        {NEG_RR,     {"sub",   true, false, false, false, false, false}},
        {NOTI_RR,    {"xori",  true, false, false, false, false, false}},
        {SLT_RRR,    {"slt",   true, false, false, false, false, false}},
        {SLTU_RRR,   {"sltu",  true, false, false, false, false, false}},
        {XORI1_RR,   {"xori",  true, false, false, false, false, false}},
        {SEQZ_RR,    {"seqz",  true, false, false, false, false, false}},
        {SNEZ_RR,    {"snez",  true, false, false, false, false, false}},
        {JAL_ZERO,   {"j",     false, false, false, true, false, false}},
        {BNEZ,       {"bnez",  false, false, false, true, false, false}},
        {CALL_SYM,   {"call",  false, false, false, false, true, false}},
        {RET,        {"ret",   false, false, false, true, false, true}},
        {ADDI_SP,    {"addi",  true, false, false, false, false, false}},
        {FMOV_RR_D,  {"fmv.d", true, false, false, false, false, false}},
        {FMOV_RR_S,  {"fmv.s", true, false, false, false, false, false}},
        {FLOAD_D,    {"fld",   true, true,  false, false, false, false}},
        {FLOAD_S,    {"flw",   true, true,  false, false, false, false}},
        {FSTORE_D,   {"fsd",   false, false, true, false, false, false}},
        {FSTORE_S,   {"fsw",   false, false, true, false, false, false}},
        {FADD_D,     {"fadd.d",true, false, false, false, false, false}},
        {FSUB_D,     {"fsub.d",true, false, false, false, false, false}},
        {FMUL_D,     {"fmul.d",true, false, false, false, false, false}},
        {FDIV_D,     {"fdiv.d",true, false, false, false, false, false}},
        {FNEG_D,     {"fneg.d",true, false, false, false, false, false}},
        {FADD_S,     {"fadd.s",true, false, false, false, false, false}},
        {FSUB_S,     {"fsub.s",true, false, false, false, false, false}},
        {FMUL_S,     {"fmul.s",true, false, false, false, false, false}},
        {FDIV_S,     {"fdiv.s",true, false, false, false, false, false}},
        {FNEG_S,     {"fneg.s",true, false, false, false, false, false}},
        {FEQ_D,      {"feq.d", true, false, false, false, false, false}},
        {FLT_D,      {"flt.d", true, false, false, false, false, false}},
        {FLE_D,      {"fle.d", true, false, false, false, false, false}},
        {FEQ_S,      {"feq.s", true, false, false, false, false, false}},
        {FLT_S,      {"flt.s", true, false, false, false, false, false}},
        {FLE_S,      {"fle.s", true, false, false, false, false, false}},
        {FMV_D_X,    {"fmv.d.x", true, false, false, false, false, false}},
        {FMV_W_X,    {"fmv.w.x", true, false, false, false, false, false}},
    };
    return table.at(target_op);
}

std::vector<MachineInstr> RiscVInstrInfo::select(const MachineInstr& g, MachineFunction& mf) {
    std::vector<MachineInstr> out;
    auto push = [&](MachineInstr mi) { out.push_back(std::move(mi)); };

    switch (g.op) {
    case MOp::ADD: push(rrr(ADD_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::SUB: push(rrr(SUB_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::AND: push(rrr(AND_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::OR:  push(rrr(OR_RRR,  g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::XOR: push(rrr(XOR_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::MUL: push(rrr(MUL_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::SDIV: push(rrr(DIV_RRR,  g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::UDIV: push(rrr(DIVU_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::SREM: push(rrr(REM_RRR,  g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::UREM: push(rrr(REMU_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::SHL:  push(rrr(SLL_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::LSHR: push(rrr(SRL_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::ASHR: push(rrr(SRA_RRR, g.defs[0].vreg, g.uses[0], g.uses[1])); break;
    case MOp::NEG:  push(rrr(NEG_RR, g.defs[0].vreg, zero(), g.uses[0])); break;
    case MOp::NOT: {
        int dst = g.defs[0].vreg;
        push(MachineInstr::makeTarget(NOTI_RR).def(MOperand::Reg_(dst, LLT::I64)).use(g.uses[0])
                 .pr(MOperand::Reg_(dst, LLT::I64)).pr(g.uses[0]).pr(MOperand::ImmOp(-1)));
        break;
    }
    case MOp::FADD: case MOp::FSUB: case MOp::FMUL: case MOp::FDIV: {
        LLT t = g.defs[0].type; bool f32 = is_f32(t);
        unsigned op = g.op == MOp::FADD ? (f32 ? FADD_S : FADD_D)
                     : g.op == MOp::FSUB ? (f32 ? FSUB_S : FSUB_D)
                     : g.op == MOp::FMUL ? (f32 ? FMUL_S : FMUL_D)
                     : (f32 ? FDIV_S : FDIV_D);
        push(MachineInstr::makeTarget(op).def(MOperand::Reg_(g.defs[0].vreg, t)).use(g.uses[0]).use(g.uses[1])
                 .pr(MOperand::Reg_(g.defs[0].vreg, t)).pr(g.uses[0]).pr(g.uses[1]));
        break;
    }
    case MOp::FNEG: {
        LLT t = g.defs[0].type; bool f32 = is_f32(t);
        push(MachineInstr::makeTarget(f32 ? FNEG_S : FNEG_D).def(MOperand::Reg_(g.defs[0].vreg, t)).use(g.uses[0])
                 .pr(MOperand::Reg_(g.defs[0].vreg, t)).pr(g.uses[0]));
        break;
    }
    case MOp::ICMP: {
        int dst = g.defs[0].vreg;
        MOperand lhs = g.uses[0], rhs = g.uses[1];
        auto slt = [&](int d, MOperand a, MOperand b, unsigned op) { push(rrr(op, d, a, b)); };
        auto negate = [&](int d, int src) {
            push(MachineInstr::makeTarget(XORI1_RR).def(MOperand::Reg_(d, LLT::I1)).use(MOperand::Reg_(src, LLT::I1))
                     .pr(MOperand::Reg_(d, LLT::I1)).pr(MOperand::Reg_(src, LLT::I1)).pr(MOperand::ImmOp(1)));
        };
        switch (g.pred) {
        case CmpPred::SLT: slt(dst, lhs, rhs, SLT_RRR); break;
        case CmpPred::SGT: slt(dst, rhs, lhs, SLT_RRR); break;
        case CmpPred::ULT: slt(dst, lhs, rhs, SLTU_RRR); break;
        case CmpPred::UGT: slt(dst, rhs, lhs, SLTU_RRR); break;
        case CmpPred::SLE: { int t = mf.new_vreg(LLT::I1); slt(t, rhs, lhs, SLT_RRR); negate(dst, t); break; }
        case CmpPred::SGE: { int t = mf.new_vreg(LLT::I1); slt(t, lhs, rhs, SLT_RRR); negate(dst, t); break; }
        case CmpPred::ULE: { int t = mf.new_vreg(LLT::I1); slt(t, rhs, lhs, SLTU_RRR); negate(dst, t); break; }
        case CmpPred::UGE: { int t = mf.new_vreg(LLT::I1); slt(t, lhs, rhs, SLTU_RRR); negate(dst, t); break; }
        case CmpPred::IEQ: { int t = mf.new_vreg(LLT::I64); push(rrr(SUB_RRR, t, lhs, rhs));
                              push(MachineInstr::makeTarget(SEQZ_RR).def(MOperand::Reg_(dst, LLT::I1)).use(MOperand::Reg_(t, LLT::I64))
                                       .pr(MOperand::Reg_(dst, LLT::I1)).pr(MOperand::Reg_(t, LLT::I64))); break; }
        case CmpPred::INE: { int t = mf.new_vreg(LLT::I64); push(rrr(SUB_RRR, t, lhs, rhs));
                              push(MachineInstr::makeTarget(SNEZ_RR).def(MOperand::Reg_(dst, LLT::I1)).use(MOperand::Reg_(t, LLT::I64))
                                       .pr(MOperand::Reg_(dst, LLT::I1)).pr(MOperand::Reg_(t, LLT::I64))); break; }
        default: break;
        }
        break;
    }
    case MOp::FCMP: {
        int dst = g.defs[0].vreg;
        MOperand lhs = g.uses[0], rhs = g.uses[1];
        bool f32 = is_f32(lhs.type);
        unsigned feq = f32 ? FEQ_S : FEQ_D, flt = f32 ? FLT_S : FLT_D, fle = f32 ? FLE_S : FLE_D;
        auto emit2 = [&](unsigned op, int d, MOperand a, MOperand b) {
            push(MachineInstr::makeTarget(op).def(MOperand::Reg_(d, LLT::I1)).use(a).use(b)
                     .pr(MOperand::Reg_(d, LLT::I1)).pr(a).pr(b));
        };
        switch (g.pred) {
        case CmpPred::FEQ: emit2(feq, dst, lhs, rhs); break;
        case CmpPred::FLT: emit2(flt, dst, lhs, rhs); break;
        case CmpPred::FLE: emit2(fle, dst, lhs, rhs); break;
        case CmpPred::FGT: emit2(flt, dst, rhs, lhs); break;
        case CmpPred::FGE: emit2(fle, dst, rhs, lhs); break;
        case CmpPred::FNE: {
            int t = mf.new_vreg(LLT::I1); emit2(feq, t, lhs, rhs);
            push(MachineInstr::makeTarget(XORI1_RR).def(MOperand::Reg_(dst, LLT::I1)).use(MOperand::Reg_(t, LLT::I1))
                     .pr(MOperand::Reg_(dst, LLT::I1)).pr(MOperand::Reg_(t, LLT::I1)).pr(MOperand::ImmOp(1)));
            break;
        }
        default: break;
        }
        break;
    }
    case MOp::LOAD: {
        int dst = g.defs[0].vreg; LLT t = g.defs[0].type; bool fp = t == LLT::F32 || t == LLT::F64;
        MOperand mem = g.uses[0].kind == MOperand::FrameIdx ? MOperand::MemFrame(g.uses[0].frame_idx, t)
                                                              : MOperand::MemReg(g.uses[0].vreg, t);
        unsigned op = fp ? (is_f32(t) ? FLOAD_S : FLOAD_D)
                     : t == LLT::I64 || t == LLT::Ptr ? LOAD_D
                     : t == LLT::I32 ? LOAD_W
                     : t == LLT::I16 ? LOAD_HU : LOAD_BU;
        push(MachineInstr::makeTarget(op).def(MOperand::Reg_(dst, t)).use(mem).pr(MOperand::Reg_(dst, t)).pr(mem));
        break;
    }
    case MOp::STORE: {
        MOperand val = g.uses[1]; LLT t = val.type; bool fp = t == LLT::F32 || t == LLT::F64;
        MOperand mem = g.uses[0].kind == MOperand::FrameIdx ? MOperand::MemFrame(g.uses[0].frame_idx, t)
                                                              : MOperand::MemReg(g.uses[0].vreg, t);
        unsigned op = fp ? (is_f32(t) ? FSTORE_S : FSTORE_D)
                     : t == LLT::I64 || t == LLT::Ptr ? STORE_D
                     : t == LLT::I32 ? STORE_W
                     : t == LLT::I16 ? STORE_H : STORE_B;
        push(MachineInstr::makeTarget(op).use(mem).use(val).pr(val).pr(mem));
        break;
    }
    case MOp::FRAME_ADDR: {
        int dst = g.defs[0].vreg; int fidx = g.uses[0].frame_idx;
        push(MachineInstr::makeTarget(LEA_FRAME).def(MOperand::Reg_(dst, LLT::Ptr))
                 .pr(MOperand::Reg_(dst, LLT::Ptr)).pr(MOperand::Phys(S0, RegClass::GPR, LLT::I64)).pr(MOperand::Frame(fidx)));
        break;
    }
    case MOp::MOV_IMM: {
        int dst = g.defs[0].vreg;
        push(MachineInstr::makeTarget(LI).def(MOperand::Reg_(dst, LLT::I64)).use(g.uses[0])
                 .pr(MOperand::Reg_(dst, LLT::I64)).pr(g.uses[0]));
        break;
    }
    case MOp::FMOV_IMM: {
        int dst = g.defs[0].vreg; LLT t = g.defs[0].type; bool f32 = is_f32(t);
        int scratch = mf.new_vreg(LLT::I64);
        int64_t bits;
        if (f32) { float fv = static_cast<float>(g.uses[0].fimm); uint32_t b32; std::memcpy(&b32, &fv, 4); bits = int64_t(int32_t(b32)); }
        else     { double dv = g.uses[0].fimm; std::memcpy(&bits, &dv, 8); }
        push(MachineInstr::makeTarget(LI).def(MOperand::Reg_(scratch, LLT::I64)).use(MOperand::ImmOp(bits, LLT::I64))
                 .pr(MOperand::Reg_(scratch, LLT::I64)).pr(MOperand::ImmOp(bits, LLT::I64)));
        push(MachineInstr::makeTarget(f32 ? FMV_W_X : FMV_D_X).def(MOperand::Reg_(dst, t)).use(MOperand::Reg_(scratch, LLT::I64))
                 .pr(MOperand::Reg_(dst, t)).pr(MOperand::Reg_(scratch, LLT::I64)));
        break;
    }
    case MOp::COPY: {
        MOperand d = g.defs[0], s = g.uses[0];
        unsigned op = d.rclass == RegClass::FPR ? (is_f32(d.type) ? FMOV_RR_S : FMOV_RR_D) : MOV_RR;
        MachineInstr mi = MachineInstr::makeTarget(op);
        mi.defs.push_back(d); mi.uses.push_back(s); mi.pr(d).pr(s);
        push(mi);
        break;
    }
    case MOp::BR:
        push(MachineInstr::makeTarget(JAL_ZERO).pr(g.uses[0]));
        break;
    case MOp::CBR:
        push(MachineInstr::makeTarget(BNEZ).use(g.uses[0]).pr(g.uses[0]).pr(g.uses[1]));
        push(MachineInstr::makeTarget(JAL_ZERO).pr(g.uses[2]));
        break;
    case MOp::RET: {
        MachineInstr mi = MachineInstr::makeTarget(RET);
        mi.uses = g.uses;
        push(mi);
        break;
    }
    case MOp::CALL: {
        MachineInstr mi = MachineInstr::makeTarget(CALL_SYM);
        mi.defs = g.defs; mi.uses = g.uses;
        mi.clobbers_gpr = g.clobbers_gpr; mi.clobbers_fpr = g.clobbers_fpr;
        mi.callee = g.callee;
        mi.pr(MOperand::Glob(g.callee));
        push(mi);
        break;
    }
    default:
        break; // PHI is eliminated before selection runs
    }
    return out;
}

} // namespace codegen::target::riscv64
