#pragma once

/// codegen/target/x86_64/X86RegisterInfo.h
///
/// Physical register numbering matches the real x86-64 encoding (0=RAX,
/// 1=RCX, ..., 15=R15; 0-15=XMM0-XMM15 for FPR) — not load-bearing for text
/// emission, but keeps the numbers meaningful if binary encoding is ever
/// added later. RBP is always reserved as the frame pointer (this backend
/// never omits it — see PrologueEpilogInserter), so it's excluded from the
/// allocatable GPR set alongside RSP.

#include "../TargetInfo.h"

namespace codegen::target::x86_64 {

enum X86Reg {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3, RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8 = 8, R9 = 9, R10 = 10, R11 = 11, R12 = 12, R13 = 13, R14 = 14, R15 = 15,
};

struct X86RegisterInfo : TargetRegisterInfo {
    const std::vector<int>& allocatable(RegClass rc) const override;
    const std::vector<int>& callee_saved(RegClass rc) const override;
    const std::vector<int>& caller_saved(RegClass rc) const override;
    int sp() const override { return RSP; }
    int fp() const override { return RBP; }
    int ra() const override { return -1; } // System V: return address lives on the stack, not in a register
    std::string reg_name(RegClass rc, int physreg, LLT width = LLT::I64) const override;
    int stack_alignment() const override { return 16; }
    int word_size() const override { return 8; }
    int64_t incoming_args_base_offset() const override { return 16; }
};

} // namespace codegen::target::x86_64
