#pragma once

/// codegen/target/riscv64/RiscVInstrInfo.h
///
/// Generic MIR -> RV64GC (LP64D) instruction selection. RISC-V's
/// register-register-register instruction shape (and direct slt/feq-style
/// compare-to-register instructions) makes this selector noticeably simpler
/// than x86-64's: no read-modify-write copy step, no CL-register shift-count
/// constraint, no flags-based compare+setcc dance, and fneg is a single
/// instruction — see the ICMP/FCMP/SHL/FNEG cases in the .cpp for the
/// contrast with codegen/target/x86_64/X86InstrInfo.cpp's equivalents.
///
/// Frame offsets are materialized with `addi rd, s0, offset`, whose
/// immediate is a 12-bit signed field; functions needing a stack frame
/// larger than +/-2KB from the frame pointer aren't supported yet (same
/// class of limitation the old codegen/legacy backend had for any
/// immediate, just narrower now that spills/ABI actually work) — a known,
/// documented gap, not a silent one.

#include "../TargetInfo.h"

namespace codegen::target::riscv64 {

enum RiscVOp : unsigned {
    MOV_RR,                                   // mv rd, rs
    LI,                                       // li rd, imm   (GNU as pseudo-op; expands for any 64-bit immediate)
    LEA_FRAME,                                // addi rd, s0, offset   (offset: bare FrameIdx operand, resolved by PEI)
    LOAD_D, LOAD_W, LOAD_HU, LOAD_BU,         // ld / lw / lhu / lbu  rd, off(rs1)
    STORE_D, STORE_W, STORE_H, STORE_B,       // sd / sw / sh / sb    rs2, off(rs1)
    ADD_RRR, SUB_RRR, AND_RRR, OR_RRR, XOR_RRR, MUL_RRR, SLL_RRR, SRL_RRR, SRA_RRR,
    DIV_RRR, DIVU_RRR, REM_RRR, REMU_RRR,
    NEG_RR,                                   // sub rd, x0, rs
    NOTI_RR,                                  // xori rd, rs, -1
    SLT_RRR, SLTU_RRR,
    XORI1_RR,                                 // xori rd, rs, 1   (boolean negate)
    SEQZ_RR, SNEZ_RR,
    JAL_ZERO,                                 // j label
    BNEZ,                                     // bnez cond, label
    CALL_SYM, RET,
    ADDI_SP,                                  // addi sp, sp, imm
    FMOV_RR_D, FMOV_RR_S,                     // fmv.d / fmv.s
    FLOAD_D, FLOAD_S, FSTORE_D, FSTORE_S,
    FADD_D, FSUB_D, FMUL_D, FDIV_D, FNEG_D,
    FADD_S, FSUB_S, FMUL_S, FDIV_S, FNEG_S,
    FEQ_D, FLT_D, FLE_D, FEQ_S, FLT_S, FLE_S,
    FMV_D_X, FMV_W_X,                         // gpr -> fpr bit-pattern move
};

struct RiscVInstrInfo : TargetInstrInfo {
    std::vector<MachineInstr> select(const MachineInstr& generic, MachineFunction& mf) override;
    const MCInstrDesc& describe(unsigned target_op) const override;
    unsigned spill_load_opcode(RegClass rc) const override { return rc == RegClass::GPR ? LOAD_D : FLOAD_D; }
    unsigned spill_store_opcode(RegClass rc) const override { return rc == RegClass::GPR ? STORE_D : FSTORE_D; }
};

} // namespace codegen::target::riscv64
