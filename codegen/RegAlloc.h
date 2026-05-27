#pragma once

/// codegen/RegAlloc.h
///
/// Linear-scan register allocator for the TensorC RISC-V backend.
///
/// Allocation strategy
/// ────────────────────
/// Physical registers are split into three banks:
///
///   GPR temporaries:  x10-x17 (a0-a7)   — 8 GPR registers
///   FPR temporaries:  f10-f17 (fa0-fa7)  — 8 FPR registers
///   Vector registers: v8-v23             — 16 vector registers
///
/// The allocator makes two passes:
///   1. Collect all virtual registers, record their RegClass.
///   2. Assign physical registers in first-seen order; spill to the stack
///      when a bank is exhausted.
///
/// Spilled operands are annotated with MachineOperand::spill(slot) and
/// RegAlloc inserts ld/sd instructions around each use/def.
///
/// Limitations (prototype)
/// ────────────────────────
/// - No liveness analysis; treats each vreg as live from first to last use.
/// - No register coalescing; the  mv a0, <vreg>  before ret may be redundant
///   if vreg was already assigned a0, but it is correct.
/// - Caller-saved / callee-saved split is not modelled; all temporaries are
///   treated as caller-saved.

#include "MachineFunction.h"

namespace codegen {

struct RegAlloc {
    void allocate(MachineFunction& mf);

    static int get_stack_offset(int spill_slot) {
        return -8 * (spill_slot + 1); // SP-relative byte offset
    }

private:
    /// Physical register banks.
    static constexpr int kGPRBase    = 10; // a0..a7
    static constexpr int kGPRCount   = 8;
    static constexpr int kFPRBase    = 10; // fa0..fa7 (FPR namespace)
    static constexpr int kFPRCount   = 8;
    static constexpr int kVecBase    = 8;  // v8..v23
    static constexpr int kVecCount   = 16;
};

} // namespace codegen