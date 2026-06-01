#include "AsmPrinter.h"
#include <sstream>

/// codegen/AsmPrinter.cpp
///
/// Emits GNU assembler syntax for RISC-V 64-bit (RV64GV).
///
/// Register naming with ABI aliases (use_abi_names = true):
///   x0  = zero   x1  = ra    x2  = sp    x3  = gp    x4  = tp
///   x5  = t0     x6  = t1    x7  = t2
///   x8  = s0/fp  x9  = s1
///   x10 = a0     x11 = a1    x12 = a2    x13 = a3    x14 = a4
///   x15 = a5     x16 = a6    x17 = a7
///   x18-x27 = s2-s11   x28-x31 = t3-t6
///   f10-f17 = fa0-fa7
///   v8-v23  = named by number

namespace codegen {

static const char* kGPRABI[] = {
    "zero","ra","sp","gp","tp","t0","t1","t2",
    "s0",  "s1","a0","a1","a2","a3","a4","a5",
    "a6",  "a7","s2","s3","s4","s5","s6","s7",
    "s8",  "s9","s10","s11","t3","t4","t5","t6"
};

std::string AsmPrinter::gpr_name(int r) const {
    if (use_abi_names && r >= 0 && r < 32)
        return kGPRABI[r];
    return "x" + std::to_string(r);
}

std::string AsmPrinter::fpr_name(int r) const {
    if (use_abi_names) {
        if (r >= 10 && r <= 17) return "fa" + std::to_string(r - 10);
        if (r >= 0  && r <= 7)  return "ft" + std::to_string(r);
    }
    return "f" + std::to_string(r);
}

std::string AsmPrinter::vec_name(int r) const {
    return "v" + std::to_string(r);
}

std::string AsmPrinter::operand_str(const MachineOperand& op) const {
    if (op.is_label) {
        return op.label;
    }
    if (op.is_mem) {
        return std::to_string(op.imm) + "(" + gpr_name(op.base_reg) + ")";
    }
    if (!op.is_reg) {
        return std::to_string(op.imm);
    }
    if (op.reg == SPILLED) {
        // Stack slot — should have been resolved by RegAlloc; emit as fallback
        int offset = RegAlloc::get_stack_offset(op.spill_slot);
        return std::to_string(offset) + "(sp)";
    }
    switch (op.rclass) {
    case RegClass::FPR:    return fpr_name(op.reg);
    case RegClass::Vector: return vec_name(op.reg);
    default:               return gpr_name(op.reg);
    }
}

// In AsmPrinter.cpp, before the instruction emission loop:
static void peephole(MachineFunction& mf)
{
    for (auto& bb : mf.blocks) {
        std::vector<MachineInstr> out;
        out.reserve(bb.instrs.size());
        
        for (size_t i = 0; i < bb.instrs.size(); ++i) {
            const auto& mi = bb.instrs[i];
            
            // Rule 1: drop identity mv (mv rX, rX)
            if (mi.opcode == "mv" && mi.ops.size() == 2
                && mi.ops[0].is_reg && mi.ops[1].is_reg
                && !mi.ops[0].is_vreg && !mi.ops[1].is_vreg
                && mi.ops[0].reg == mi.ops[1].reg) {
                continue;
            }
            
            // Rule 2: drop mv rA, rT followed immediately by mv rT, rA (round-trip)
            if (mi.opcode == "mv" && mi.ops.size() == 2
                && i + 1 < bb.instrs.size()) {
                const auto& next = bb.instrs[i + 1];
                if (next.opcode == "mv" && next.ops.size() == 2
                    && mi.ops[0].is_reg && !mi.ops[0].is_vreg
                    && mi.ops[1].is_reg && !mi.ops[1].is_vreg
                    && next.ops[0].is_reg && !next.ops[0].is_vreg
                    && next.ops[1].is_reg && !next.ops[1].is_vreg
                    && mi.ops[0].reg == next.ops[1].reg
                    && mi.ops[1].reg == next.ops[0].reg) {
                    // mv rA, rT; mv rT, rA — skip both
                    ++i;
                    continue;
                }
            }
            
            out.push_back(mi);
        }
        bb.instrs = std::move(out);
    }
}

void AsmPrinter::print(const MachineFunction& mf)
{
    // Function prologue
    os << "\t.text\n";
    os << "\t.globl\t" << mf.name << "\n";
    os << "\t.type\t" << mf.name << ", @function\n";
    os << mf.name << ":\n";

    // Stack frame: allocate space for spill slots if any
    if (mf.spill_slots > 0) {
        int frame_size = 8 * mf.spill_slots;
        // Align to 16 bytes
        frame_size = (frame_size + 15) & ~15;
        os << "\taddi\tsp, sp, -" << frame_size << "\n";
    }

    peephole(const_cast<MachineFunction&>(mf)); // apply peephole optimizations before printing
    for (const auto& bb : mf.blocks) {
        os << bb.label << ":\n";
        for (const auto& mi : bb.instrs) {
            if (mi.is_pseudo) continue; // skip pseudo-instructions
            if (mi.opcode.empty()) continue; // skip blank placeholders

            // Comments (lines starting with #)
            if (!mi.opcode.empty() && mi.opcode[0] == '#') {
                os << "\t" << mi.opcode << "\n";
                continue;
            }

            os << "\t" << mi.opcode;

            for (size_t i = 0; i < mi.ops.size(); ++i) {
                os << (i == 0 ? "\t" : ", ");
                os << operand_str(mi.ops[i]);
            }
            os << "\n";
        }
    }

    // Stack frame: restore SP if spill slots were used
    if (mf.spill_slots > 0) {
        int frame_size = 8 * mf.spill_slots;
        frame_size = (frame_size + 15) & ~15;
        os << "\taddi\tsp, sp, " << frame_size << "\n";
    }

    os << "\t.size\t" << mf.name << ", .-" << mf.name << "\n";
}

} // namespace codegen
