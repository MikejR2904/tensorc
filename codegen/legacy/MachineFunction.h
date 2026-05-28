#pragma once

/// codegen/MachineFunction.h
///
/// Container types for machine-level IR.
///
/// VRegCounter
/// ───────────
/// InstrSelector allocates virtual register IDs via VRegCounter::next().
/// This gives each IR Value a stable, unique small integer — never a
/// truncated pointer, which collides on 64-bit builds.

#include "MachineInstr.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace codegen {

/// Monotonically increasing virtual register allocator.
/// One instance lives on MachineFunction; InstrSelector receives a pointer.
struct VRegCounter {
    int next() { return counter_++; }
    int count() const { return counter_; }
private:
    int counter_ = 0;
};

struct MachineBasicBlock {
    std::string              label;
    std::vector<MachineInstr> instrs;
};

struct MachineFunction {
    std::string                    name;
    std::vector<MachineBasicBlock> blocks;

    /// Virtual register allocator — all passes use this to stay consistent.
    VRegCounter vregs;

    /// Maps IR Value pointers to virtual register IDs so InstrSelector can
    /// reuse the same vreg when the same Value appears in multiple instructions.
    std::unordered_map<const void*, int> value_to_vreg;
    std::unordered_map<const void*, std::string> block_labels;

    /// Allocate or look up the vreg for an IR Value.
    int vreg_for(const void* val) {
        auto it = value_to_vreg.find(val);
        if (it != value_to_vreg.end()) return it->second;
        int id = vregs.next();
        value_to_vreg[val] = id;
        return id;
    }

    std::string label_for(const void* block) const {
        auto it = block_labels.find(block);
        return it == block_labels.end() ? "entry" : it->second;
    }

    /// Total number of spill slots allocated by RegAlloc.
    int spill_slots = 0;
};

} // namespace codegen
