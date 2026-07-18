#pragma once

/// codegen/target/riscv64/RiscVRegisterInfo.h
///
/// RV64GC, LP64D ABI. Physical register numbers match the real ISA encoding
/// (x0-x31, f0-f31). x8 (s0/fp) is always reserved as the frame pointer,
/// same "never omit FP" simplification as the x86-64 backend.

#include "../TargetInfo.h"

namespace codegen::target::riscv64 {

enum RvReg {
    X0 = 0, RA = 1, SP = 2, GP = 3, TP = 4, T0 = 5, T1 = 6, T2 = 7,
    S0 = 8, S1 = 9, A0 = 10, A1 = 11, A2 = 12, A3 = 13, A4 = 14, A5 = 15, A6 = 16, A7 = 17,
    S2 = 18, S3 = 19, S4 = 20, S5 = 21, S6 = 22, S7 = 23, S8 = 24, S9 = 25, S10 = 26, S11 = 27,
    T3 = 28, T4 = 29, T5 = 30, T6 = 31,
};

struct RiscVRegisterInfo : TargetRegisterInfo {
    const std::vector<int>& allocatable(RegClass rc) const override;
    const std::vector<int>& callee_saved(RegClass rc) const override;
    const std::vector<int>& caller_saved(RegClass rc) const override;
    int sp() const override { return SP; }
    int fp() const override { return S0; }
    int ra() const override { return RA; }
    std::string reg_name(RegClass rc, int physreg, LLT width = LLT::I64) const override;
    int stack_alignment() const override { return 16; }
    int word_size() const override { return 8; }
    int64_t incoming_args_base_offset() const override { return 0; }
};

} // namespace codegen::target::riscv64
