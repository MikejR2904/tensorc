#pragma once

/// codegen/mir/PrologueEpilogInserter.h ("PEI")
///
/// Runs last, after RegAlloc has decided which callee-saved registers got
/// used and how many spill slots exist. Computes the final stack frame
/// layout (target-independent arithmetic — see .cpp), resolves every
/// Mem/FrameIdx operand in the function to a concrete FP-relative
/// displacement, and splices in the real prologue/epilogue via the target's
/// FrameLowering. Replaces codegen/legacy/AsmPrinter's prologue logic, which
/// only ever accounts for `spill_slots` — no callee-saved save/restore, no
/// frame pointer chain, no incoming/outgoing stack argument space at all.

#include "MachineIR.h"
#include "../target/TargetInfo.h"

namespace codegen::mir {

void insert_prologue_epilogue(MachineFunction& mf, const target::TargetRegisterInfo& tri,
                               const target::TargetInstrInfo& tii, target::FrameLowering& fl);

} // namespace codegen::mir
