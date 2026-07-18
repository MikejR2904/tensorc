#include "RiscVAsmPrinter.h"
#include <cassert>

namespace codegen::target::riscv64 {

namespace {

std::string fmt_operand(const MOperand& o, const TargetRegisterInfo& tri) {
    switch (o.kind) {
        case MOperand::Reg:
            return tri.reg_name(o.rclass, o.vreg, o.type);
        case MOperand::Imm:
            return std::to_string(o.imm);
        case MOperand::FImm:
            return std::to_string(static_cast<int64_t>(o.fimm)); // not expected post-selection
        case MOperand::Mem: {
            assert(o.frame_idx < 0 && "frame-relative Mem operand must be resolved by PrologueEpilogInserter before printing");
            return std::to_string(o.imm) + "(" + tri.reg_name(RegClass::GPR, o.vreg, LLT::I64) + ")";
        }
        case MOperand::Global:
            return o.global;
        case MOperand::Block:
            return o.block->label;
        case MOperand::FrameIdx:
            return "<unresolved-frame-idx>";
    }
    return "?";
}

} // namespace

void RiscVAsmPrinter::print(std::ostream& os, const MachineFunction& mf, const TargetInstrInfo& tii, const TargetRegisterInfo& tri) {
    os << "\t.text\n\t.globl " << mf.name << "\n" << mf.name << ":\n";
    for (auto& bb : mf.blocks) {
        os << bb->label << ":\n";
        for (auto& mi : bb->instrs) {
            assert(!mi.is_generic && "AsmPrinter requires fully target-selected MIR");
            const MCInstrDesc& desc = tii.describe(mi.target_op);
            os << "\t" << desc.mnemonic;
            for (size_t i = 0; i < mi.print_operands.size(); ++i)
                os << (i == 0 ? "\t" : ", ") << fmt_operand(mi.print_operands[i], tri);
            os << "\n";
        }
    }
}

} // namespace codegen::target::riscv64
