#pragma once

/// codegen/mir/Liveness.h
///
/// Real liveness analysis over target-selected MIR — this is the single
/// biggest correctness gap the redesign closes. The old codegen/legacy
/// register allocator (codegen/legacy/RegAlloc.cpp) treats every virtual
/// register as live for the *entire function*, which is why
/// test_legacy_reg_pressure.s spills three temporaries a properly
/// liveness-aware allocator wouldn't need to.
///
/// Standard two-step construction (Poletto-Sarkar): backward per-block
/// dataflow first (so live ranges follow real CFG reachability — this
/// matters because PhiElimination.cpp gives merged variables *multiple*
/// definition sites, one per predecessor edge, so naively scanning
/// "first def to last use in layout order" would either under- or
/// over-shoot badly), then flatten into one [start,end] interval per vreg
/// over a single block-layout-order instruction numbering. Collapsing to one
/// contiguous range per vreg (rather than tracking lifetime holes) is the
/// standard "fast" linear-scan simplification — always safe (never
/// under-approximates a live range, so never double-books a register),
/// occasionally spills more than a hole-aware allocator would.

#include "MachineIR.h"
#include <unordered_map>
#include <vector>

namespace codegen::mir {

struct LiveInterval {
    int vreg;
    RegClass rclass;
    int start; // instruction index (block-layout order) of first liveness
    int end;   // instruction index of last liveness
};

struct LivenessResult {
    /// Instruction index (0-based, block-layout order) of each block's first
    /// and one-past-last instruction — used to extend intervals across
    /// blocks where a vreg is merely live-through (live-in and live-out but
    /// neither defined nor used there).
    std::unordered_map<const MachineBasicBlock*, std::pair<int, int>> block_range;
    /// One interval per vreg that appears in the function, sorted by start.
    std::vector<LiveInterval> intervals;
};

LivenessResult compute_liveness(const MachineFunction& mf);

} // namespace codegen::mir
