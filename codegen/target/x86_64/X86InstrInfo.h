#pragma once

/// codegen/target/x86_64/X86InstrInfo.h
///
/// Generic MIR -> x86-64 (System V, AT&T syntax) instruction selection.
/// Every opcode here is fully width-qualified (e.g. separate LOAD_L/LOAD_Q
/// entries rather than one LOAD_MEM opcode disambiguated at print time) so
/// AsmPrinter can stay a dumb "look up mnemonic, print print_operands" loop
/// with no per-instruction special-casing.
///
/// No immediate folding into instruction operands is implemented (every
/// constant was already materialized into its own vreg by GenericISel) —
/// this is the -O0-equivalent baseline described in the backend redesign
/// plan; folding immediates into e.g. `addq $imm, dst` is a straightforward
/// peephole for later, not required for correctness.

#include "../TargetInfo.h"

namespace codegen::target::x86_64 {

enum X86Op : unsigned {
    MOV_RR,                                   // movq   src, dst
    MOV_RI32,                                 // movl   $imm, dst32   (zero-extends to 64 bits)
    MOVABS_RI64,                              // movabsq $imm, dst
    LEA_FRAME,                                // leaq   off(%rbp), dst   (frame_idx resolved by PEI)
    LOAD_Q, LOAD_L, LOAD_W, LOAD_B,           // movq / movl / movzwq / movzbq  mem, dst
    STORE_Q, STORE_L, STORE_W, STORE_B,       // movq / movl / movw   / movb    src, mem
    ADD_RR, SUB_RR, AND_RR, OR_RR, XOR_RR, IMUL_RR, // 2-operand RMW: op src, dst
    NEG_R, NOT_R,
    SHL_CL, SHR_CL, SAR_CL,                   // shl/shr/sar %cl, dst
    ZERO_R,                                   // xorl dst32, dst32
    CQTO,                                     // sign-extend rax -> rdx:rax   (64-bit divide)
    CDQ,                                      // sign-extend eax -> edx:eax   (32-bit divide)
    IDIV_R, DIV_R,                            // idivq / divq divisor  (reads/writes rdx:rax)
    IDIV_R32, DIV_R32,                        // idivl / divl divisor  (reads/writes edx:eax)
    CMP_RR,                                   // cmpq src2, src1
    TEST_RR,                                  // testq src, src
    SETCC,                                    // setCC dst8   (.pred selects the condition)
    JMP, JCC,                                 // jmp label ; jCC label (.pred selects the condition)
    CALL_SYM, RET,
    PUSH_R, POP_R,
    SUB_RSP_IMM, ADD_RSP_IMM,
    FMOV_RR_SD, FMOV_RR_SS,
    FLOAD_SD, FLOAD_SS, FSTORE_SD, FSTORE_SS,
    FADD_SD, FSUB_SD, FMUL_SD, FDIV_SD,
    FADD_SS, FSUB_SS, FMUL_SS, FDIV_SS,
    FXOR_PD, FXOR_PS,                         // sign-bit flip for FNEG
    FCMP_SD, FCMP_SS,                         // ucomisd / ucomiss
    GPR_TO_FPR_Q, GPR_TO_FPR_D,               // movq / movd  gpr, xmm  (bit-pattern move)
};

struct X86InstrInfo : TargetInstrInfo {
    std::vector<MachineInstr> select(const MachineInstr& generic, MachineFunction& mf) override;
    const MCInstrDesc& describe(unsigned target_op) const override;
    unsigned spill_load_opcode(RegClass rc) const override { return rc == RegClass::GPR ? LOAD_Q : FLOAD_SD; }
    unsigned spill_store_opcode(RegClass rc) const override { return rc == RegClass::GPR ? STORE_Q : FSTORE_SD; }
};

} // namespace codegen::target::x86_64
