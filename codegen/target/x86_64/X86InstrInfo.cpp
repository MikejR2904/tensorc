#include "X86InstrInfo.h"
#include "X86RegisterInfo.h"
#include <cstring>
#include <unordered_map>

namespace codegen::target::x86_64 {

namespace {
bool fits_i32(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }
bool is_f32(LLT t) { return t == LLT::F32; }

/// Every bool (I1)-producing instruction elsewhere in this file (ICMP/FCMP's
/// ZERO_R-then-SETCC sequence, LOAD_B's movzbq, CallLowering's ABI copies)
/// zero-extends the *entire* physical register, not just the low byte — that
/// invariant is what lets CBR's TEST_RR safely test the full 64-bit register
/// (see its own comment). But `&&`/`||`/`!` on bool operands lower through
/// this same GPR AND/OR/XOR/NOT machinery as bitwise ops on ints (TensorC's
/// frontend uses one BinOpCode::And for both, disambiguated only by operand
/// type), and if width-selection follows the logical I1 type literally, the
/// RMW instruction becomes an 8-bit write (`and %dl,%al`) — which x86 does
/// *not* auto zero-extend, unlike a 32-bit write. Whatever garbage
/// previously occupied bits 8-63 of that physical register (e.g. a stale
/// parameter value) survives untouched, so a later `testq` on the "boolean"
/// can read nonzero even when the real 0/1 result is 0. Operating at 32-bit
/// width instead — safe precisely because the operands are already
/// guaranteed zero-extended — sidesteps this entirely and is simpler than
/// re-zeroing before every logical op.
LLT machine_int_width(LLT logical) { return logical == LLT::I1 ? LLT::I32 : logical; }

/// Same operand, width coerced via machine_int_width — needed on *every*
/// register operand of an integer RMW instruction, source included: fixing
/// only the destination's width (as an earlier version of this function did)
/// still leaves e.g. `mov %sil, %eax` — an 8-bit SETCC result copied
/// straight in as `src` while `dst` got coerced to 32-bit — mismatched.
MOperand as_machine_width(MOperand o) {
    if (o.kind == MOperand::Reg) o.type = machine_int_width(o.type);
    return o;
}

/// AT&T-syntax 2-operand read-modify-write: `dst <- lhs` then `op rhs, dst`.
/// `t` must be the *destination's* width (I32 or I64) — AT&T `mov`/RMW
/// instructions have no explicit size suffix here, so the width comes
/// entirely from which register-name alias gets printed (e.g. %eax vs
/// %rax), and both operands of a single instruction must agree or GNU `as`
/// rejects it ("operand type mismatch"). Using a fixed I64 regardless of the
/// actual operand type was a real, previously-uncaught bug: it happened to
/// print harmlessly-matching 64-bit names when every value on the path was
/// i64, but broke (mismatched widths) the moment an i32 value came from
/// memory or a 32-bit register.
MachineInstr movrr(int dst, MOperand src, LLT t) {
    return MachineInstr::makeTarget(MOV_RR).def(MOperand::Reg_(dst, t)).use(src)
                                            .pr(src).pr(MOperand::Reg_(dst, t));
}
MachineInstr fmovrr(int dst, MOperand src, LLT t) {
    unsigned op = is_f32(t) ? FMOV_RR_SS : FMOV_RR_SD;
    return MachineInstr::makeTarget(op).def(MOperand::Reg_(dst, t)).use(src).pr(src).pr(MOperand::Reg_(dst, t));
}
} // namespace

const MCInstrDesc& X86InstrInfo::describe(unsigned target_op) const {
    static const std::unordered_map<unsigned, MCInstrDesc> table = {
        {MOV_RR,       {"mov",     true, false, false, false, false, false}},
        {MOV_RI32,     {"movl",    true, false, false, false, false, false}},
        {MOVABS_RI64,  {"movabsq", true, false, false, false, false, false}},
        {LEA_FRAME,    {"leaq",    true, false, false, false, false, false}},
        {LOAD_Q,       {"movq",    true, true,  false, false, false, false}},
        {LOAD_L,       {"movl",    true, true,  false, false, false, false}},
        {LOAD_W,       {"movzwq",  true, true,  false, false, false, false}},
        {LOAD_B,       {"movzbq",  true, true,  false, false, false, false}},
        {STORE_Q,      {"movq",    false, false, true, false, false, false}},
        {STORE_L,      {"movl",    false, false, true, false, false, false}},
        {STORE_W,      {"movw",    false, false, true, false, false, false}},
        {STORE_B,      {"movb",    false, false, true, false, false, false}},
        {ADD_RR,       {"add",     true, false, false, false, false, false}},
        {SUB_RR,       {"sub",     true, false, false, false, false, false}},
        {AND_RR,       {"and",     true, false, false, false, false, false}},
        {OR_RR,        {"or",      true, false, false, false, false, false}},
        {XOR_RR,       {"xor",     true, false, false, false, false, false}},
        {IMUL_RR,      {"imul",    true, false, false, false, false, false}},
        {NEG_R,        {"neg",     true, false, false, false, false, false}},
        {NOT_R,        {"not",     true, false, false, false, false, false}},
        {SHL_CL,       {"shl",     true, false, false, false, false, false}},
        {SHR_CL,       {"shr",     true, false, false, false, false, false}},
        {SAR_CL,       {"sar",     true, false, false, false, false, false}},
        {ZERO_R,       {"xorl",    true, false, false, false, false, false}},
        {CQTO,         {"cqto",    true, false, false, false, false, false}},
        {CDQ,          {"cltd",    true, false, false, false, false, false}},
        {IDIV_R,       {"idivq",   true, false, false, false, false, false}},
        {DIV_R,        {"divq",    true, false, false, false, false, false}},
        {IDIV_R32,     {"idivl",   true, false, false, false, false, false}},
        {DIV_R32,      {"divl",    true, false, false, false, false, false}},
        {CMP_RR,       {"cmp",     false, false, false, false, false, false}},
        {TEST_RR,      {"testq",   false, false, false, false, false, false}},
        {SETCC,        {"set",     true, false, false, false, false, false}},
        {JMP,          {"jmp",     false, false, false, true, false, false}},
        {JCC,          {"j",       false, false, false, true, false, false}},
        {CALL_SYM,     {"call",    false, false, false, false, true, false}},
        {RET,          {"ret",     false, false, false, true, false, true}},
        {PUSH_R,       {"push",    false, false, true, false, false, false}},
        {POP_R,        {"pop",     true, false, false, false, false, false}},
        {SUB_RSP_IMM,  {"sub",     true, false, false, false, false, false}},
        {ADD_RSP_IMM,  {"add",     true, false, false, false, false, false}},
        {FMOV_RR_SD,   {"movsd",   true, false, false, false, false, false}},
        {FMOV_RR_SS,   {"movss",   true, false, false, false, false, false}},
        {FLOAD_SD,     {"movsd",   true, true,  false, false, false, false}},
        {FLOAD_SS,     {"movss",   true, true,  false, false, false, false}},
        {FSTORE_SD,    {"movsd",   false, false, true, false, false, false}},
        {FSTORE_SS,    {"movss",   false, false, true, false, false, false}},
        {FADD_SD,      {"addsd",   true, false, false, false, false, false}},
        {FSUB_SD,      {"subsd",   true, false, false, false, false, false}},
        {FMUL_SD,      {"mulsd",   true, false, false, false, false, false}},
        {FDIV_SD,      {"divsd",   true, false, false, false, false, false}},
        {FADD_SS,      {"addss",   true, false, false, false, false, false}},
        {FSUB_SS,      {"subss",   true, false, false, false, false, false}},
        {FMUL_SS,      {"mulss",   true, false, false, false, false, false}},
        {FDIV_SS,      {"divss",   true, false, false, false, false, false}},
        {FXOR_PD,      {"xorpd",   true, false, false, false, false, false}},
        {FXOR_PS,      {"xorps",   true, false, false, false, false, false}},
        {FCMP_SD,      {"ucomisd", false, false, false, false, false, false}},
        {FCMP_SS,      {"ucomiss", false, false, false, false, false, false}},
        {GPR_TO_FPR_Q, {"movq",    true, false, false, false, false, false}},
        {GPR_TO_FPR_D, {"movd",    true, false, false, false, false, false}},
    };
    return table.at(target_op);
}

std::vector<MachineInstr> X86InstrInfo::select(const MachineInstr& g, MachineFunction& mf) {
    std::vector<MachineInstr> out;
    auto push = [&](MachineInstr mi) { out.push_back(std::move(mi)); };

    switch (g.op) {
    case MOp::ADD: case MOp::SUB: case MOp::AND: case MOp::OR: case MOp::XOR: case MOp::MUL: {
        int dst = g.defs[0].vreg;
        LLT t = machine_int_width(g.defs[0].type);
        unsigned rmw = g.op == MOp::ADD ? ADD_RR : g.op == MOp::SUB ? SUB_RR : g.op == MOp::AND ? AND_RR
                      : g.op == MOp::OR ? OR_RR : g.op == MOp::XOR ? XOR_RR : IMUL_RR;
        push(movrr(dst, as_machine_width(g.uses[0]), t));
        push(MachineInstr::makeTarget(rmw).def(MOperand::Reg_(dst, t)).use(MOperand::Reg_(dst, t)).use(as_machine_width(g.uses[1]))
                 .pr(as_machine_width(g.uses[1])).pr(MOperand::Reg_(dst, t)));
        break;
    }
    case MOp::SDIV: case MOp::SREM: case MOp::UDIV: case MOp::UREM: {
        int dst = g.defs[0].vreg;
        LLT t = g.defs[0].type;
        bool is64 = (t == LLT::I64 || t == LLT::Ptr);
        bool is_signed = (g.op == MOp::SDIV || g.op == MOp::SREM);
        bool want_rem = (g.op == MOp::SREM || g.op == MOp::UREM);
        push(MachineInstr::makeTarget(MOV_RR).def(MOperand::Phys(RAX, RegClass::GPR, t)).use(g.uses[0])
                 .pr(g.uses[0]).pr(MOperand::Phys(RAX, RegClass::GPR, t)));
        if (is_signed) {
            push(MachineInstr::makeTarget(is64 ? CQTO : CDQ).def(MOperand::Phys(RDX, RegClass::GPR, t))
                     .use(MOperand::Phys(RAX, RegClass::GPR, t)));
        } else {
            push(MachineInstr::makeTarget(ZERO_R).def(MOperand::Phys(RDX, RegClass::GPR, LLT::I32))
                     .pr(MOperand::Phys(RDX, RegClass::GPR, LLT::I32)).pr(MOperand::Phys(RDX, RegClass::GPR, LLT::I32)));
        }
        unsigned div_op = is_signed ? (is64 ? IDIV_R : IDIV_R32) : (is64 ? DIV_R : DIV_R32);
        push(MachineInstr::makeTarget(div_op)
                 .def(MOperand::Phys(RAX, RegClass::GPR, t)).def(MOperand::Phys(RDX, RegClass::GPR, t))
                 .use(MOperand::Phys(RAX, RegClass::GPR, t)).use(MOperand::Phys(RDX, RegClass::GPR, t)).use(g.uses[1])
                 .pr(g.uses[1]));
        int result_phys = want_rem ? RDX : RAX;
        push(MachineInstr::makeTarget(MOV_RR).def(MOperand::Reg_(dst, t)).use(MOperand::Phys(result_phys, RegClass::GPR, t))
                 .pr(MOperand::Phys(result_phys, RegClass::GPR, t)).pr(MOperand::Reg_(dst, t)));
        break;
    }
    case MOp::SHL: case MOp::LSHR: case MOp::ASHR: {
        int dst = g.defs[0].vreg;
        LLT t = machine_int_width(g.defs[0].type);
        unsigned sop = g.op == MOp::SHL ? SHL_CL : g.op == MOp::LSHR ? SHR_CL : SAR_CL;
        push(movrr(dst, as_machine_width(g.uses[0]), t));
        LLT count_t = machine_int_width(g.uses[1].type);
        push(MachineInstr::makeTarget(MOV_RR).def(MOperand::Phys(RCX, RegClass::GPR, count_t)).use(g.uses[1])
                 .pr(as_machine_width(g.uses[1])).pr(MOperand::Phys(RCX, RegClass::GPR, count_t)));
        push(MachineInstr::makeTarget(sop).def(MOperand::Reg_(dst, t))
                 .use(MOperand::Reg_(dst, t)).use(MOperand::Phys(RCX, RegClass::GPR, LLT::I8))
                 .pr(MOperand::Phys(RCX, RegClass::GPR, LLT::I8)).pr(MOperand::Reg_(dst, t)));
        break;
    }
    case MOp::NEG: case MOp::NOT: {
        int dst = g.defs[0].vreg;
        LLT t = machine_int_width(g.defs[0].type);
        push(movrr(dst, as_machine_width(g.uses[0]), t));
        push(MachineInstr::makeTarget(g.op == MOp::NEG ? NEG_R : NOT_R)
                 .def(MOperand::Reg_(dst, t)).use(MOperand::Reg_(dst, t)).pr(MOperand::Reg_(dst, t)));
        break;
    }
    case MOp::FADD: case MOp::FSUB: case MOp::FMUL: case MOp::FDIV: {
        int dst = g.defs[0].vreg;
        LLT t = g.defs[0].type;
        bool f32 = is_f32(t);
        unsigned rmw = g.op == MOp::FADD ? (f32 ? FADD_SS : FADD_SD)
                      : g.op == MOp::FSUB ? (f32 ? FSUB_SS : FSUB_SD)
                      : g.op == MOp::FMUL ? (f32 ? FMUL_SS : FMUL_SD)
                      : (f32 ? FDIV_SS : FDIV_SD);
        push(fmovrr(dst, g.uses[0], t));
        push(MachineInstr::makeTarget(rmw).def(MOperand::Reg_(dst, t)).use(MOperand::Reg_(dst, t)).use(g.uses[1])
                 .pr(g.uses[1]).pr(MOperand::Reg_(dst, t)));
        break;
    }
    case MOp::FNEG: {
        int dst = g.defs[0].vreg;
        LLT t = g.defs[0].type;
        bool f32 = is_f32(t);
        push(fmovrr(dst, g.uses[0], t));
        int scratch_gpr = mf.new_vreg(f32 ? LLT::I32 : LLT::I64);
        int64_t sign_bit = f32 ? int64_t(int32_t(0x80000000u)) : int64_t(INT64_MIN);
        if (f32) {
            push(MachineInstr::makeTarget(MOV_RI32).def(MOperand::Reg_(scratch_gpr, LLT::I32)).use(MOperand::ImmOp(sign_bit, LLT::I32))
                     .pr(MOperand::ImmOp(sign_bit, LLT::I32)).pr(MOperand::Reg_(scratch_gpr, LLT::I32)));
        } else {
            push(MachineInstr::makeTarget(MOVABS_RI64).def(MOperand::Reg_(scratch_gpr, LLT::I64)).use(MOperand::ImmOp(sign_bit, LLT::I64))
                     .pr(MOperand::ImmOp(sign_bit, LLT::I64)).pr(MOperand::Reg_(scratch_gpr, LLT::I64)));
        }
        int scratch_fpr = mf.new_vreg(t);
        push(MachineInstr::makeTarget(f32 ? GPR_TO_FPR_D : GPR_TO_FPR_Q).def(MOperand::Reg_(scratch_fpr, t)).use(MOperand::Reg_(scratch_gpr, f32 ? LLT::I32 : LLT::I64))
                 .pr(MOperand::Reg_(scratch_gpr, f32 ? LLT::I32 : LLT::I64)).pr(MOperand::Reg_(scratch_fpr, t)));
        push(MachineInstr::makeTarget(f32 ? FXOR_PS : FXOR_PD).def(MOperand::Reg_(dst, t)).use(MOperand::Reg_(dst, t)).use(MOperand::Reg_(scratch_fpr, t))
                 .pr(MOperand::Reg_(scratch_fpr, t)).pr(MOperand::Reg_(dst, t)));
        break;
    }
    case MOp::ICMP: case MOp::FCMP: {
        int dst = g.defs[0].vreg;
        bool fp = g.op == MOp::FCMP;
        push(MachineInstr::makeTarget(ZERO_R).def(MOperand::Reg_(dst, LLT::I32))
                 .pr(MOperand::Reg_(dst, LLT::I32)).pr(MOperand::Reg_(dst, LLT::I32)));
        LLT operand_t = g.uses[0].type;
        unsigned cmp_op = fp ? (is_f32(operand_t) ? FCMP_SS : FCMP_SD) : CMP_RR;
        push(MachineInstr::makeTarget(cmp_op).use(g.uses[0]).use(g.uses[1]).pr(g.uses[1]).pr(g.uses[0]));
        MachineInstr setcc = MachineInstr::makeTarget(SETCC);
        setcc.pred = g.pred;
        setcc.def(MOperand::Reg_(dst, LLT::I1)).pr(MOperand::Reg_(dst, LLT::I1));
        push(setcc);
        break;
    }
    case MOp::LOAD: {
        int dst = g.defs[0].vreg;
        LLT t = g.defs[0].type;
        bool fp = t == LLT::F32 || t == LLT::F64;
        MOperand mem = g.uses[0].kind == MOperand::FrameIdx ? MOperand::MemFrame(g.uses[0].frame_idx, t)
                                                              : MOperand::MemReg(g.uses[0].vreg, t);
        unsigned op = fp ? (is_f32(t) ? FLOAD_SS : FLOAD_SD)
                     : t == LLT::I64 || t == LLT::Ptr ? LOAD_Q
                     : t == LLT::I32 ? LOAD_L
                     : t == LLT::I16 ? LOAD_W : LOAD_B;
        MachineInstr mi = MachineInstr::makeTarget(op).def(MOperand::Reg_(dst, t)).use(mem).pr(mem).pr(MOperand::Reg_(dst, t));
        push(mi);
        break;
    }
    case MOp::STORE: {
        MOperand val = g.uses[1];
        LLT t = val.type;
        bool fp = t == LLT::F32 || t == LLT::F64;
        MOperand mem = g.uses[0].kind == MOperand::FrameIdx ? MOperand::MemFrame(g.uses[0].frame_idx, t)
                                                              : MOperand::MemReg(g.uses[0].vreg, t);
        unsigned op = fp ? (is_f32(t) ? FSTORE_SS : FSTORE_SD)
                     : t == LLT::I64 || t == LLT::Ptr ? STORE_Q
                     : t == LLT::I32 ? STORE_L
                     : t == LLT::I16 ? STORE_W : STORE_B;
        MachineInstr mi = MachineInstr::makeTarget(op).use(mem).use(val).pr(val).pr(mem);
        push(mi);
        break;
    }
    case MOp::FRAME_ADDR: {
        int dst = g.defs[0].vreg;
        int fidx = g.uses[0].frame_idx;
        push(MachineInstr::makeTarget(LEA_FRAME).def(MOperand::Reg_(dst, LLT::Ptr))
                 .pr(MOperand::MemFrame(fidx, LLT::Ptr)).pr(MOperand::Reg_(dst, LLT::Ptr)));
        break;
    }
    case MOp::MOV_IMM: {
        int dst = g.defs[0].vreg;
        int64_t v = g.uses[0].imm;
        if (fits_i32(v)) {
            push(MachineInstr::makeTarget(MOV_RI32).def(MOperand::Reg_(dst, LLT::I32)).use(MOperand::ImmOp(v, LLT::I32))
                     .pr(MOperand::ImmOp(v, LLT::I32)).pr(MOperand::Reg_(dst, LLT::I32)));
        } else {
            push(MachineInstr::makeTarget(MOVABS_RI64).def(MOperand::Reg_(dst, LLT::I64)).use(MOperand::ImmOp(v, LLT::I64))
                     .pr(MOperand::ImmOp(v, LLT::I64)).pr(MOperand::Reg_(dst, LLT::I64)));
        }
        break;
    }
    case MOp::FMOV_IMM: {
        int dst = g.defs[0].vreg;
        LLT t = g.defs[0].type;
        bool f32 = is_f32(t);
        int scratch = mf.new_vreg(f32 ? LLT::I32 : LLT::I64);
        if (f32) {
            float fv = static_cast<float>(g.uses[0].fimm);
            uint32_t bits; std::memcpy(&bits, &fv, 4);
            int64_t as_i32 = int64_t(int32_t(bits));
            push(MachineInstr::makeTarget(MOV_RI32).def(MOperand::Reg_(scratch, LLT::I32)).use(MOperand::ImmOp(as_i32, LLT::I32))
                     .pr(MOperand::ImmOp(as_i32, LLT::I32)).pr(MOperand::Reg_(scratch, LLT::I32)));
            push(MachineInstr::makeTarget(GPR_TO_FPR_D).def(MOperand::Reg_(dst, t)).use(MOperand::Reg_(scratch, LLT::I32))
                     .pr(MOperand::Reg_(scratch, LLT::I32)).pr(MOperand::Reg_(dst, t)));
        } else {
            double dv = g.uses[0].fimm;
            int64_t bits; std::memcpy(&bits, &dv, 8);
            push(MachineInstr::makeTarget(MOVABS_RI64).def(MOperand::Reg_(scratch, LLT::I64)).use(MOperand::ImmOp(bits, LLT::I64))
                     .pr(MOperand::ImmOp(bits, LLT::I64)).pr(MOperand::Reg_(scratch, LLT::I64)));
            push(MachineInstr::makeTarget(GPR_TO_FPR_Q).def(MOperand::Reg_(dst, t)).use(MOperand::Reg_(scratch, LLT::I64))
                     .pr(MOperand::Reg_(scratch, LLT::I64)).pr(MOperand::Reg_(dst, t)));
        }
        break;
    }
    case MOp::COPY: {
        // Built directly (not via the movrr/fmovrr helpers, which always
        // construct a fresh *virtual* Reg_ destination from a bare vreg id
        // — wrong here, since `d` may be a hard-pinned physical register
        // from CallLowering's ABI copies, e.g. def=[Phys(xmm0)] for a float
        // return value; going through the helpers silently dropped
        // is_physical and let RegAlloc reassign it, corrupting the ABI).
        MOperand d = g.defs[0], s = g.uses[0];
        unsigned op = d.rclass == RegClass::FPR ? (is_f32(d.type) ? FMOV_RR_SS : FMOV_RR_SD) : MOV_RR;
        MachineInstr mi = MachineInstr::makeTarget(op);
        mi.defs.push_back(d); mi.uses.push_back(s); mi.pr(s).pr(d);
        push(mi);
        break;
    }
    case MOp::BR: {
        push(MachineInstr::makeTarget(JMP).pr(g.uses[0]));
        break;
    }
    case MOp::CBR: {
        // testq needs a 64-bit register alias printed even though the
        // boolean's logical type is I1 — every producer of a boolean vreg
        // (ICMP's setcc-after-zeroing, LOAD_B's movzbq, a raw bool param)
        // zero-extends the full register, so testing all 64 bits is safe.
        MOperand cond64 = g.uses[0]; cond64.type = LLT::I64;
        push(MachineInstr::makeTarget(TEST_RR).use(g.uses[0]).use(g.uses[0]).pr(cond64).pr(cond64));
        MachineInstr jcc = MachineInstr::makeTarget(JCC);
        jcc.pred = CmpPred::INE;
        jcc.pr(g.uses[1]);
        push(jcc);
        push(MachineInstr::makeTarget(JMP).pr(g.uses[2]));
        break;
    }
    case MOp::RET: {
        MachineInstr mi = MachineInstr::makeTarget(RET);
        mi.uses = g.uses; // carried-forward Phys(return-reg) use from CallLowering, if any
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
        break; // PHI is eliminated before selection runs; nothing else reaches here
    }
    return out;
}

} // namespace codegen::target::x86_64
