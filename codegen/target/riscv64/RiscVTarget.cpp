#include "RiscVTarget.h"
#include "RiscVRegisterInfo.h"
#include "RiscVInstrInfo.h"
#include "RiscVAsmPrinter.h"

namespace codegen::target::riscv64 {

namespace {

/// RV64 LP64D ABI: first 8 integer/pointer args in a0-a7; first 8 float args
/// in fa0-fa7 (float args do NOT consume integer registers, and vice versa,
/// per the actual LP64D rule for register-passable floats). Overflow spills
/// to the stack. Return in a0 (int/ptr) or fa0 (float).
struct RiscVCallingConv : CallingConv {
    std::vector<ArgLocation> classify_args(const std::vector<LLT>& arg_types) const override {
        int int_idx = 0, float_idx = 0;
        int64_t stack_off = 0;
        std::vector<ArgLocation> out;
        for (LLT t : arg_types) {
            ArgLocation loc;
            if (mir::is_float_llt(t)) {
                if (float_idx < 8) { loc.in_register = true; loc.physreg = 10 + float_idx; loc.rclass = RegClass::FPR; float_idx++; } // fa0-fa7 = f10-f17
                else { loc.in_register = false; loc.stack_offset = stack_off; stack_off += 8; }
            } else {
                if (int_idx < 8) { loc.in_register = true; loc.physreg = A0 + int_idx; loc.rclass = RegClass::GPR; int_idx++; }
                else { loc.in_register = false; loc.stack_offset = stack_off; stack_off += 8; }
            }
            out.push_back(loc);
        }
        return out;
    }

    ArgLocation classify_return(LLT ret_type) const override {
        ArgLocation loc; loc.in_register = true;
        if (mir::is_float_llt(ret_type)) { loc.physreg = 10; loc.rclass = RegClass::FPR; } // fa0 (f10)
        else { loc.physreg = A0; loc.rclass = RegClass::GPR; }
        return loc;
    }
};

/// RISC-V frame shape: `addi sp,sp,-N` then `addi s0,sp,N` (s0 now points at
/// the pre-call SP, i.e. the top of this frame) then explicit saves of ra
/// (always — RISC-V has no auto-pushed return address, unlike x86-64's
/// `call`) and any used callee-saved registers, mirrored in reverse for the
/// epilogue.
struct RiscVFrameLowering : FrameLowering {
    std::vector<MachineInstr> emit_prologue(const FrameLayout& layout) const override {
        std::vector<MachineInstr> out;
        out.push_back(MachineInstr::makeTarget(ADDI_SP).def(MOperand::Phys(SP, RegClass::GPR, LLT::I64)).use(MOperand::Phys(SP, RegClass::GPR, LLT::I64))
                          .pr(MOperand::Phys(SP, RegClass::GPR, LLT::I64)).pr(MOperand::Phys(SP, RegClass::GPR, LLT::I64)).pr(MOperand::ImmOp(-layout.frame_size)));
        out.push_back(MachineInstr::makeTarget(LEA_FRAME).def(MOperand::Phys(S0, RegClass::GPR, LLT::I64)).use(MOperand::Phys(SP, RegClass::GPR, LLT::I64))
                          .pr(MOperand::Phys(S0, RegClass::GPR, LLT::I64)).pr(MOperand::Phys(SP, RegClass::GPR, LLT::I64)).pr(MOperand::ImmOp(layout.frame_size)));
        if (layout.save_ra) {
            out.push_back(MachineInstr::makeTarget(STORE_D).use(MOperand::MemPhys(S0, LLT::I64, layout.ra_offset)).use(MOperand::Phys(RA, RegClass::GPR, LLT::I64))
                              .pr(MOperand::Phys(RA, RegClass::GPR, LLT::I64)).pr(MOperand::MemPhys(S0, LLT::I64, layout.ra_offset)));
        }
        for (auto& s : layout.saved_callee_regs) {
            unsigned op = s.rclass == RegClass::GPR ? STORE_D : FSTORE_D;
            out.push_back(MachineInstr::makeTarget(op).use(MOperand::MemPhys(S0, LLT::I64, s.offset)).use(MOperand::Phys(s.physreg, s.rclass, LLT::I64))
                              .pr(MOperand::Phys(s.physreg, s.rclass, LLT::I64)).pr(MOperand::MemPhys(S0, LLT::I64, s.offset)));
        }
        return out;
    }
    std::vector<MachineInstr> emit_epilogue(const FrameLayout& layout) const override {
        std::vector<MachineInstr> out;
        for (auto& s : layout.saved_callee_regs) {
            unsigned op = s.rclass == RegClass::GPR ? LOAD_D : FLOAD_D;
            out.push_back(MachineInstr::makeTarget(op).def(MOperand::Phys(s.physreg, s.rclass, LLT::I64)).use(MOperand::MemPhys(S0, LLT::I64, s.offset))
                              .pr(MOperand::Phys(s.physreg, s.rclass, LLT::I64)).pr(MOperand::MemPhys(S0, LLT::I64, s.offset)));
        }
        if (layout.save_ra) {
            out.push_back(MachineInstr::makeTarget(LOAD_D).def(MOperand::Phys(RA, RegClass::GPR, LLT::I64)).use(MOperand::MemPhys(S0, LLT::I64, layout.ra_offset))
                              .pr(MOperand::Phys(RA, RegClass::GPR, LLT::I64)).pr(MOperand::MemPhys(S0, LLT::I64, layout.ra_offset)));
        }
        out.push_back(MachineInstr::makeTarget(MOV_RR).def(MOperand::Phys(SP, RegClass::GPR, LLT::I64)).use(MOperand::Phys(S0, RegClass::GPR, LLT::I64))
                          .pr(MOperand::Phys(SP, RegClass::GPR, LLT::I64)).pr(MOperand::Phys(S0, RegClass::GPR, LLT::I64)));
        return out;
    }
};

struct RiscVTarget : Target {
    RiscVRegisterInfo reg_info_;
    RiscVInstrInfo instr_info_;
    RiscVCallingConv cc_;
    RiscVFrameLowering frame_;
    std::unique_ptr<TargetAsmPrinter> printer_;

    RiscVTarget() : printer_(std::make_unique<RiscVAsmPrinter>()) {}

    const char* name() const override { return "riscv64"; }
    TargetInstrInfo& instr_info() override { return instr_info_; }
    TargetRegisterInfo& reg_info() override { return reg_info_; }
    CallingConv& calling_conv() override { return cc_; }
    TargetAsmPrinter& asm_printer() override { return *printer_; }
    FrameLowering& frame_lowering() override { return frame_; }
};

} // namespace

std::unique_ptr<Target> create_riscv64_target() { return std::make_unique<RiscVTarget>(); }

} // namespace codegen::target::riscv64
