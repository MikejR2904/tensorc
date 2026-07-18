#include "X86RegisterInfo.h"
#include <array>

namespace codegen::target::x86_64 {

const std::vector<int>& X86RegisterInfo::allocatable(RegClass rc) const {
    // Caller-saved first (cheapest — no save/restore needed unless live
    // across a call, which LinearScanRegAlloc accounts for via CALL's
    // clobber list), callee-saved after. RSP/RBP excluded (RBP is the
    // permanent frame pointer; RSP is the stack pointer).
    static const std::vector<int> gpr = {RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, RBX, R12, R13, R14, R15};
    static const std::vector<int> fpr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}; // xmm0-xmm15
    return rc == RegClass::GPR ? gpr : fpr;
}

const std::vector<int>& X86RegisterInfo::callee_saved(RegClass rc) const {
    static const std::vector<int> gpr = {RBX, R12, R13, R14, R15};
    static const std::vector<int> fpr = {}; // SysV AMD64: no callee-saved XMM registers
    return rc == RegClass::GPR ? gpr : fpr;
}

const std::vector<int>& X86RegisterInfo::caller_saved(RegClass rc) const {
    static const std::vector<int> gpr = {RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11};
    static const std::vector<int> fpr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    return rc == RegClass::GPR ? gpr : fpr;
}

std::string X86RegisterInfo::reg_name(RegClass rc, int physreg, LLT width) const {
    if (rc == RegClass::FPR) return "xmm" + std::to_string(physreg);

    static const char* q[16] = {"rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi","r8","r9","r10","r11","r12","r13","r14","r15"};
    static const char* d[16] = {"eax","ecx","edx","ebx","esp","ebp","esi","edi","r8d","r9d","r10d","r11d","r12d","r13d","r14d","r15d"};
    static const char* w[16] = {"ax","cx","dx","bx","sp","bp","si","di","r8w","r9w","r10w","r11w","r12w","r13w","r14w","r15w"};
    static const char* b[16] = {"al","cl","dl","bl","spl","bpl","sil","dil","r8b","r9b","r10b","r11b","r12b","r13b","r14b","r15b"};

    if (physreg < 0 || physreg > 15) return "?";
    switch (width) {
        case LLT::I1: case LLT::I8:  return b[physreg];
        case LLT::I16:                return w[physreg];
        case LLT::I32:                return d[physreg];
        default:                      return q[physreg]; // I64 / F32 / F64 / Ptr (FPR handled above)
    }
}

} // namespace codegen::target::x86_64
