#include "X86Target.h"
#include "X86RegisterInfo.h"
#include "X86InstrInfo.h"
#include "X86AsmPrinter.h"
#include <cassert>

namespace codegen::target::x86_64 {

namespace {

/// System V AMD64 ABI: first 6 integer/pointer args in RDI,RSI,RDX,RCX,R8,R9;
/// first 8 float args in XMM0-XMM7; overflow goes on the stack. Return value
/// in RAX (int/ptr) or XMM0 (float).
struct X86CallingConv : CallingConv {
    std::vector<ArgLocation> classify_args(const std::vector<LLT>& arg_types) const override {
        static const int int_regs[] = {RDI, RSI, RDX, RCX, R8, R9};
        int int_idx = 0, float_idx = 0;
        int64_t stack_off = 0;
        std::vector<ArgLocation> out;
        for (LLT t : arg_types) {
            ArgLocation loc;
            if (mir::is_float_llt(t)) {
                if (float_idx < 8) { loc.in_register = true; loc.physreg = float_idx++; loc.rclass = RegClass::FPR; }
                else { loc.in_register = false; loc.stack_offset = stack_off; stack_off += 8; }
            } else {
                if (int_idx < 6) { loc.in_register = true; loc.physreg = int_regs[int_idx++]; loc.rclass = RegClass::GPR; }
                else { loc.in_register = false; loc.stack_offset = stack_off; stack_off += 8; }
            }
            out.push_back(loc);
        }
        return out;
    }

    ArgLocation classify_return(LLT ret_type) const override {
        ArgLocation loc;
        loc.in_register = true;
        if (mir::is_float_llt(ret_type)) { loc.physreg = 0; loc.rclass = RegClass::FPR; } // xmm0
        else { loc.physreg = RAX; loc.rclass = RegClass::GPR; }
        return loc;
    }
};

/// x86-64 frame shape: `push %rbp; mov %rsp,%rbp; sub $N,%rsp` then explicit
/// stores for any other used callee-saved GPRs/XMMs, mirrored in reverse for
/// the epilogue. The return address needs no explicit save/restore — `call`
/// pushes it and `ret` pops it automatically.
struct X86FrameLowering : FrameLowering {
    std::vector<MachineInstr> emit_prologue(const FrameLayout& layout) const override {
        std::vector<MachineInstr> out;
        out.push_back(MachineInstr::makeTarget(PUSH_R).use(MOperand::Phys(RBP, RegClass::GPR, LLT::I64)).pr(MOperand::Phys(RBP, RegClass::GPR, LLT::I64)));
        out.push_back(MachineInstr::makeTarget(MOV_RR).def(MOperand::Phys(RBP, RegClass::GPR, LLT::I64)).use(MOperand::Phys(RSP, RegClass::GPR, LLT::I64))
                          .pr(MOperand::Phys(RSP, RegClass::GPR, LLT::I64)).pr(MOperand::Phys(RBP, RegClass::GPR, LLT::I64)));
        if (layout.frame_size > 0) {
            out.push_back(MachineInstr::makeTarget(SUB_RSP_IMM).def(MOperand::Phys(RSP, RegClass::GPR, LLT::I64))
                              .pr(MOperand::ImmOp(layout.frame_size)).pr(MOperand::Phys(RSP, RegClass::GPR, LLT::I64)));
        }
        for (auto& s : layout.saved_callee_regs) {
            unsigned op = s.rclass == RegClass::GPR ? STORE_Q : FSTORE_SD;
            out.push_back(MachineInstr::makeTarget(op).use(MOperand::MemPhys(RBP, LLT::I64, s.offset)).use(MOperand::Phys(s.physreg, s.rclass, LLT::I64))
                              .pr(MOperand::Phys(s.physreg, s.rclass, LLT::I64)).pr(MOperand::MemPhys(RBP, LLT::I64, s.offset)));
        }
        return out;
    }
    std::vector<MachineInstr> emit_epilogue(const FrameLayout& layout) const override {
        std::vector<MachineInstr> out;
        for (auto& s : layout.saved_callee_regs) {
            unsigned op = s.rclass == RegClass::GPR ? LOAD_Q : FLOAD_SD;
            out.push_back(MachineInstr::makeTarget(op).def(MOperand::Phys(s.physreg, s.rclass, LLT::I64)).use(MOperand::MemPhys(RBP, LLT::I64, s.offset))
                              .pr(MOperand::MemPhys(RBP, LLT::I64, s.offset)).pr(MOperand::Phys(s.physreg, s.rclass, LLT::I64)));
        }
        out.push_back(MachineInstr::makeTarget(MOV_RR).def(MOperand::Phys(RSP, RegClass::GPR, LLT::I64)).use(MOperand::Phys(RBP, RegClass::GPR, LLT::I64))
                          .pr(MOperand::Phys(RBP, RegClass::GPR, LLT::I64)).pr(MOperand::Phys(RSP, RegClass::GPR, LLT::I64)));
        out.push_back(MachineInstr::makeTarget(POP_R).def(MOperand::Phys(RBP, RegClass::GPR, LLT::I64)).pr(MOperand::Phys(RBP, RegClass::GPR, LLT::I64)));
        return out;
    }
};

struct X86Target : Target {
    X86RegisterInfo reg_info_;
    X86InstrInfo instr_info_;
    X86CallingConv cc_;
    X86FrameLowering frame_;
    std::unique_ptr<TargetAsmPrinter> printer_;

    X86Target() : printer_(std::make_unique<X86AsmPrinter>()) {}

    const char* name() const override { return "x86_64"; }
    TargetInstrInfo& instr_info() override { return instr_info_; }
    TargetRegisterInfo& reg_info() override { return reg_info_; }
    CallingConv& calling_conv() override { return cc_; }
    TargetAsmPrinter& asm_printer() override { return *printer_; }
    FrameLowering& frame_lowering() override { return frame_; }
};

} // namespace

std::unique_ptr<Target> create_x86_64_target() { return std::make_unique<X86Target>(); }

} // namespace codegen::target::x86_64
