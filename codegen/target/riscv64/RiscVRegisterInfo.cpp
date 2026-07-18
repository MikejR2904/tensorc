#include "RiscVRegisterInfo.h"

namespace codegen::target::riscv64 {

const std::vector<int>& RiscVRegisterInfo::allocatable(RegClass rc) const {
    // Caller-saved (temporaries + arg regs) first, callee-saved after.
    // x0(zero), ra, sp, gp, tp, s0(fp) excluded.
    static const std::vector<int> gpr = {T0, T1, T2, A0, A1, A2, A3, A4, A5, A6, A7, T3, T4, T5, T6,
                                          S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11};
    static const std::vector<int> fpr = {0,1,2,3,4,5,6,7, 10,11,12,13,14,15,16,17, 28,29,30,31,
                                          8,9, 18,19,20,21,22,23,24,25,26,27};
    return rc == RegClass::GPR ? gpr : fpr;
}

const std::vector<int>& RiscVRegisterInfo::callee_saved(RegClass rc) const {
    static const std::vector<int> gpr = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11}; // s0 handled separately as the fixed FP
    static const std::vector<int> fpr = {8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}; // fs0-fs11
    return rc == RegClass::GPR ? gpr : fpr;
}

const std::vector<int>& RiscVRegisterInfo::caller_saved(RegClass rc) const {
    static const std::vector<int> gpr = {T0, T1, T2, A0, A1, A2, A3, A4, A5, A6, A7, T3, T4, T5, T6};
    static const std::vector<int> fpr = {0,1,2,3,4,5,6,7, 10,11,12,13,14,15,16,17, 28,29,30,31};
    return rc == RegClass::GPR ? gpr : fpr;
}

std::string RiscVRegisterInfo::reg_name(RegClass rc, int physreg, LLT /*width*/) const {
    if (rc == RegClass::FPR) {
        static const char* f[32] = {
            "ft0","ft1","ft2","ft3","ft4","ft5","ft6","ft7","fs0","fs1",
            "fa0","fa1","fa2","fa3","fa4","fa5","fa6","fa7",
            "fs2","fs3","fs4","fs5","fs6","fs7","fs8","fs9","fs10","fs11",
            "ft8","ft9","ft10","ft11"};
        return (physreg >= 0 && physreg < 32) ? f[physreg] : "?";
    }
    static const char* x[32] = {
        "zero","ra","sp","gp","tp","t0","t1","t2","s0","s1",
        "a0","a1","a2","a3","a4","a5","a6","a7",
        "s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","t3","t4","t5","t6"};
    return (physreg >= 0 && physreg < 32) ? x[physreg] : "?";
}

} // namespace codegen::target::riscv64
