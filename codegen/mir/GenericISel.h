#pragma once

/// codegen/mir/GenericISel.h
///
/// Lowers one ir::Function into a target-independent MachineFunction.
/// Implements ir::InstructionVisitor so dispatch needs no dynamic_cast
/// (same pattern as the old codegen/legacy/InstrSelector). Unlike that
/// selector, this never bakes in a calling convention or a target mnemonic:
/// CallInst/ReturnInst become generic CALL/RET (codegen/mir/CallLowering.h
/// assigns ABI locations afterwards), and every arithmetic/compare op becomes
/// a generic MOp (codegen/target/*/*.cpp assigns real opcodes afterwards).
///
/// TensorOpInst and the async/cast/reshape instruction family are out of
/// scope here, same as the legacy selector (which also left them as no-ops)
/// — tensor-op lowering is a separate, untouched pipeline
/// (codegen/bridge/TensorOpLowering.*).

#include "MachineIR.h"
#include "../../compiler/ir/Instruction.h"
#include "../../compiler/ir/IRModule.h"
#include <unordered_map>

namespace codegen::mir {

class GenericISel : public ir::InstructionVisitor {
public:
    explicit GenericISel(MachineFunction& mf) : mf_(mf) {}

    /// Lower every block of `fn` into mf_. Requires the frontend's CFGPass to
    /// have already populated ir::BasicBlock::preds/succs (done by
    /// ir::PassPipeline::run()) — MIR blocks reuse those edges directly
    /// instead of recomputing a CFG.
    void run(ir::Function& fn);

    void visit(ir::BinOpInst&) override;
    void visit(ir::UnOpInst&) override;
    void visit(ir::CmpInst&) override;
    void visit(ir::AllocaInst&) override;
    void visit(ir::LoadInst&) override;
    void visit(ir::StoreInst&) override;
    void visit(ir::BranchInst&) override;
    void visit(ir::CondBranchInst&) override;
    void visit(ir::ReturnInst&) override;
    void visit(ir::CallInst&) override;
    void visit(ir::PhiInst&) override;

private:
    MachineFunction& mf_;
    MachineBasicBlock* cur_ = nullptr;
    std::unordered_map<const void*, int> value_to_vreg_;
    std::unordered_map<const void*, MachineBasicBlock*> block_map_;
    std::unordered_map<const void*, int> alloca_frame_index_;

    static LLT llt_of(const TypePtr&);
    int vreg_for(ir::Value* v);
    /// Materialize an ir::Value operand. Constants always become a fresh
    /// vreg defined by MOV_IMM/FMOV_IMM — immediate folding into a target
    /// instruction's operand list is a selection-time peephole, deliberately
    /// not implemented in this phase (see codegen/target/*/​*InstrInfo.cpp).
    MOperand operand_for(ir::Value* v);
    /// Like operand_for, but never emits an instruction: constants stay
    /// literal Imm/FImm operands. Used only for PHI incoming values, whose
    /// constant sources must be materialized in the *predecessor* block by
    /// PhiElimination, not in the block containing the phi (see GenericISel.cpp).
    MOperand raw_value_operand(ir::Value* v);
    void emit(MachineInstr mi) { cur_->instrs.push_back(std::move(mi)); }
};

} // namespace codegen::mir
