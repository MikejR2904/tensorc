#pragma once

/// codegen/mir/MIROperands.h
///
/// Shared operand-walking helpers used by both Liveness.cpp and
/// LinearScanRegAlloc.cpp, so the two agree on exactly what counts as a
/// "virtual register occurrence" in a target-selected MachineInstr — a
/// Reg-kind operand that isn't already physical, or the base register of a
/// register-relative Mem operand (frame_idx == -1, also not physical).

#include "MachineIR.h"
#include <functional>

namespace codegen::mir {

inline bool is_vreg_operand(const MOperand& o) {
    if (o.is_physical) return false;
    if (o.kind == MOperand::Reg) return true;
    if (o.kind == MOperand::Mem && o.frame_idx < 0) return true; // register-relative base
    return false;
}

/// Visits every def-position vreg occurrence in `mi` (defs list only).
inline void for_each_def_vreg(MachineInstr& mi, const std::function<void(MOperand&)>& fn) {
    for (auto& o : mi.defs) if (is_vreg_operand(o)) fn(o);
}
inline void for_each_def_vreg(const MachineInstr& mi, const std::function<void(const MOperand&)>& fn) {
    for (auto& o : mi.defs) if (is_vreg_operand(o)) fn(o);
}

/// Visits every use-position vreg occurrence in `mi` (uses list only; a
/// register-relative Mem operand inside `uses` counts as a use of its base).
inline void for_each_use_vreg(MachineInstr& mi, const std::function<void(MOperand&)>& fn) {
    for (auto& o : mi.uses) if (is_vreg_operand(o)) fn(o);
}
inline void for_each_use_vreg(const MachineInstr& mi, const std::function<void(const MOperand&)>& fn) {
    for (auto& o : mi.uses) if (is_vreg_operand(o)) fn(o);
}

/// Visits every vreg occurrence across defs, uses, AND print_operands — used
/// by RegAlloc's rewrite pass, which must keep all three lists consistent
/// (see MachineInstr::print_operands' invariant comment in MachineIR.h).
inline void for_each_vreg_operand(MachineInstr& mi, const std::function<void(MOperand&)>& fn) {
    for (auto& o : mi.defs) if (is_vreg_operand(o)) fn(o);
    for (auto& o : mi.uses) if (is_vreg_operand(o)) fn(o);
    for (auto& o : mi.print_operands) if (is_vreg_operand(o)) fn(o);
}

} // namespace codegen::mir
