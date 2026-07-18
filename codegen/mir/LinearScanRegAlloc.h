#pragma once

/// codegen/mir/LinearScanRegAlloc.h
///
/// Poletto-Sarkar linear-scan allocation over the real liveness computed by
/// Liveness.h — replaces codegen/legacy/RegAlloc.cpp's greedy "assign vregs
/// in ID order, spill when the bank runs out" allocator, which has no
/// liveness analysis at all (see that file's own doc comment). Also
/// replaces codegen/legacy/RegAllocGraphColoring.cpp, a from-scratch
/// Chaitin's-algorithm implementation that was never wired into the CMake
/// build and doesn't compile as written (see the backend redesign plan) —
/// linear scan is a smaller, well-understood step up from "no liveness" that
/// is actually correct and shipping, rather than a more sophisticated
/// allocator that only exists on paper.
///
/// Runs entirely through TargetRegisterInfo/TargetInstrInfo (allocatable
/// registers, callee-saved sets, spill opcodes) — no architecture-specific
/// logic lives here, so this one implementation serves every target.

#include "MachineIR.h"
#include "../target/TargetInfo.h"

namespace codegen::mir {

struct LinearScanRegAlloc {
    /// Mutates mf in place: rewrites every vreg operand to a physical
    /// register or, when a class's registers are exhausted for an
    /// interval's live range, inserts spill-load/spill-store instructions
    /// around each def/use and creates a frame_object per spilled vreg.
    /// Populates mf.used_callee_saved_{gpr,fpr} for PrologueEpilogInserter.
    void allocate(MachineFunction& mf, const target::TargetRegisterInfo& tri, const target::TargetInstrInfo& tii);
};

} // namespace codegen::mir
