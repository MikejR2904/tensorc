#pragma once

/// codegen/InstrSelector.h
///
/// InstrSelector — lowers IR instructions to MachineInstr sequences.
///
/// Implements InstructionVisitor so it dispatches without dynamic_cast chains.
/// Each visit() method:
///   1. Looks up or allocates virtual register IDs via mf->vreg_for(ptr).
///   2. Emits MachineInstr(s) into the current MachineBasicBlock.
///   3. Uses MachineOperand factory helpers (vreg/preg/imm_op) for clarity.
///
/// After InstrSelector runs, all operands hold virtual register IDs.
/// RegAlloc::allocate() then rewrites them to physical registers or spill slots.

#include "../compiler/ir/Instruction.h"
#include "MachineFunction.h"

namespace codegen {

struct InstrSelector : ir::InstructionVisitor {
    explicit InstrSelector(MachineFunction* mf) : mf(mf) {}

    void start_block(std::string label) {
        mf->blocks.push_back({std::move(label), {}});
    }

    // ── Visitor overrides ─────────────────────────────────────────────────
    void visit(ir::BinOpInst&)      override;
    void visit(ir::UnOpInst&)       override;
    void visit(ir::CmpInst&)        override;
    void visit(ir::LoadInst&)       override;
    void visit(ir::StoreInst&)      override;
    void visit(ir::TensorOpInst&)   override;
    void visit(ir::CallInst&)       override;
    void visit(ir::ReturnInst&)     override;
    void visit(ir::PhiInst&)        override;
    void visit(ir::BranchInst&)     override;
    void visit(ir::CondBranchInst&) override;
    // Remaining visitors use visit_default() no-op from base

private:
    MachineFunction* mf;

    /// Get or create the vreg for an IR value pointer.
    int vreg(const void* val) { return mf->vreg_for(val); }

    /// Emit into the current (last) basic block, creating "entry" if needed.
    MachineBasicBlock& current_bb() {
        if (mf->blocks.empty()) mf->blocks.push_back({"entry", {}});
        return mf->blocks.back();
    }

    void emit(MachineInstr mi) { current_bb().instrs.push_back(std::move(mi)); }
};

} // namespace codegen
