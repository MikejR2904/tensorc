#pragma once

/// codegen/MachineInstr.h
///
/// Machine-level instruction representation for the TensorC RISC-V backend.
///
/// Virtual register encoding
/// ─────────────────────────
/// MachineOperand::reg holds a virtual register ID when is_reg==true and the
/// instruction is freshly selected.  After RegAlloc::allocate():
///   reg >= 0        → physical register number (e.g. 10 == x10/a0)
///   reg == SPILLED  → operand was spilled; AsmPrinter emits stack-slot access
///
/// The is_vreg flag distinguishes pre-allocation (virtual) from post-allocation
/// (physical) operands so passes can assert on the expected state.

#include <string>
#include <vector>
#include <cstdint>

namespace codegen {

enum class RegClass { GPR, FPR, Vector };

/// Sentinel: operand was spill-allocated.  AsmPrinter uses spill_slot to compute
/// the stack offset.
static constexpr int SPILLED = INT32_MIN;

struct MachineOperand {
    bool    is_reg    = false; // true → register operand (virtual or physical)
    bool    is_vreg   = true;  // true before RegAlloc; false after (physical)
    int     reg       = -1;    // virtual / physical register number
    int     spill_slot = -1;   // valid only when reg == SPILLED after RegAlloc
    int64_t imm       = 0;     // immediate value (when is_reg == false)
    RegClass rclass   = RegClass::GPR;

    // ── Factory helpers ───────────────────────────────────────────────────

    static MachineOperand vreg(int id, RegClass rc = RegClass::GPR) {
        MachineOperand op; op.is_reg = true; op.is_vreg = true;
        op.reg = id; op.rclass = rc; return op;
    }

    static MachineOperand preg(int id, RegClass rc = RegClass::GPR) {
        MachineOperand op; op.is_reg = true; op.is_vreg = false;
        op.reg = id; op.rclass = rc; return op;
    }

    static MachineOperand imm_op(int64_t v) {
        MachineOperand op; op.is_reg = false; op.imm = v; return op;
    }

    static MachineOperand spill(int slot, RegClass rc = RegClass::GPR) {
        MachineOperand op; op.is_reg = true; op.is_vreg = false;
        op.reg = SPILLED; op.spill_slot = slot; op.rclass = rc; return op;
    }
};

struct MachineInstr {
    std::string               opcode;
    std::vector<MachineOperand> ops;

    // True if this is a pseudo-instruction with no assembly output
    // (e.g. the COPY pseudo used during phi lowering).
    bool is_pseudo = false;

    // ── Convenience constructors ──────────────────────────────────────────

    static MachineInstr make(std::string opc) {
        MachineInstr mi; mi.opcode = std::move(opc); return mi;
    }

    MachineInstr& add(MachineOperand op) { ops.push_back(op); return *this; }
};

} // namespace codegen
