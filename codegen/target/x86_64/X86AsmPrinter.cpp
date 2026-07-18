#include "X86AsmPrinter.h"
#include "X86InstrInfo.h"
#include <cassert>

namespace codegen::target::x86_64 {

namespace {

const char* cond_suffix(CmpPred p) {
    switch (p) {
        case CmpPred::IEQ: case CmpPred::FEQ: return "e";
        case CmpPred::INE: case CmpPred::FNE: return "ne";
        case CmpPred::SLT: return "l";  case CmpPred::SLE: return "le";
        case CmpPred::SGT: return "g";  case CmpPred::SGE: return "ge";
        case CmpPred::ULT: case CmpPred::FLT: return "b";
        case CmpPred::ULE: case CmpPred::FLE: return "be";
        case CmpPred::UGT: case CmpPred::FGT: return "a";
        case CmpPred::UGE: case CmpPred::FGE: return "ae";
    }
    return "e";
}

std::string fmt_operand(const MOperand& o, const TargetRegisterInfo& tri) {
    switch (o.kind) {
        case MOperand::Reg:
            return "%" + tri.reg_name(o.rclass, o.vreg, o.type);
        case MOperand::Imm:
            return "$" + std::to_string(o.imm);
        case MOperand::FImm:
            return "$" + std::to_string(static_cast<int64_t>(o.fimm)); // not expected post-selection
        case MOperand::Mem: {
            assert(o.frame_idx < 0 && "frame-relative Mem operand must be resolved by PrologueEpilogInserter before printing");
            std::string disp = o.imm != 0 ? std::to_string(o.imm) : "";
            return disp + "(%" + tri.reg_name(RegClass::GPR, o.vreg, LLT::I64) + ")";
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

void X86AsmPrinter::print(std::ostream& os, const MachineFunction& mf, const TargetInstrInfo& tii, const TargetRegisterInfo& tri) {
    os << "\t.text\n\t.globl " << mf.name << "\n" << mf.name << ":\n";
    for (auto& bb : mf.blocks) {
        os << bb->label << ":\n";
        for (auto& mi : bb->instrs) {
            assert(!mi.is_generic && "AsmPrinter requires fully target-selected MIR");
            const MCInstrDesc& desc = tii.describe(mi.target_op);

            if (mi.target_op == SETCC) { os << "\tset" << cond_suffix(mi.pred) << "\t" << fmt_operand(mi.print_operands[0], tri) << "\n"; continue; }
            if (mi.target_op == JCC)   { os << "\tj" << cond_suffix(mi.pred) << "\t" << fmt_operand(mi.print_operands[0], tri) << "\n"; continue; }

            os << "\t" << desc.mnemonic;
            for (size_t i = 0; i < mi.print_operands.size(); ++i)
                os << (i == 0 ? "\t" : ", ") << fmt_operand(mi.print_operands[i], tri);
            os << "\n";
        }
    }
}

} // namespace codegen::target::x86_64
