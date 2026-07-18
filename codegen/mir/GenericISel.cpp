#include "GenericISel.h"
#include <cassert>

namespace codegen::mir {

LLT GenericISel::llt_of(const TypePtr& t) {
    if (!t) return LLT::I64;
    switch (t->kind) {
        case Type::Kind::I32:  return LLT::I32;
        case Type::Kind::I64:  return LLT::I64;
        case Type::Kind::F32:  return LLT::F32;
        case Type::Kind::F64:  return LLT::F64;
        case Type::Kind::Bool: return LLT::I1;
        default:               return LLT::Ptr; // opaque handle: str/array/tensor/etc — out of scope this phase
    }
}

int GenericISel::vreg_for(ir::Value* v) {
    auto it = value_to_vreg_.find(v);
    if (it != value_to_vreg_.end()) return it->second;
    int id = mf_.new_vreg(llt_of(v->type));
    value_to_vreg_[v] = id;
    return id;
}

MOperand GenericISel::operand_for(ir::Value* v) {
    if (auto* ci = dynamic_cast<ir::ConstantInt*>(v)) {
        LLT t = llt_of(ci->type);
        int dst = mf_.new_vreg(t);
        emit(MachineInstr::make(MOp::MOV_IMM).def(MOperand::Reg_(dst, t)).use(MOperand::ImmOp(ci->value, t)));
        return MOperand::Reg_(dst, t);
    }
    if (auto* cf = dynamic_cast<ir::ConstantFloat*>(v)) {
        LLT t = llt_of(cf->type);
        int dst = mf_.new_vreg(t);
        emit(MachineInstr::make(MOp::FMOV_IMM).def(MOperand::Reg_(dst, t)).use(MOperand::FImmOp(cf->value, t)));
        return MOperand::Reg_(dst, t);
    }
    if (auto* cb = dynamic_cast<ir::ConstantBool*>(v)) {
        int dst = mf_.new_vreg(LLT::I1);
        emit(MachineInstr::make(MOp::MOV_IMM).def(MOperand::Reg_(dst, LLT::I1)).use(MOperand::ImmOp(cb->value ? 1 : 0, LLT::I1)));
        return MOperand::Reg_(dst, LLT::I1);
    }
    // Argument or Instruction: already has (or will get) a stable vreg.
    int id = vreg_for(v);
    return MOperand::Reg_(id, llt_of(v->type));
}

MOperand GenericISel::raw_value_operand(ir::Value* v) {
    if (auto* ci = dynamic_cast<ir::ConstantInt*>(v)) return MOperand::ImmOp(ci->value, llt_of(ci->type));
    if (auto* cf = dynamic_cast<ir::ConstantFloat*>(v)) return MOperand::FImmOp(cf->value, llt_of(cf->type));
    if (auto* cb = dynamic_cast<ir::ConstantBool*>(v)) return MOperand::ImmOp(cb->value ? 1 : 0, LLT::I1);
    return MOperand::Reg_(vreg_for(v), llt_of(v->type));
}

void GenericISel::run(ir::Function& fn) {
    // Strip the IR's "@" global-symbol sigil (compiler/ir/IRBuilder.h prefixes
    // every user function name with it) so the emitted .globl/label matches
    // what CALL sites actually target — visit(CallInst&) below already
    // strips it from the callee name it calls, so leaving it on the
    // *definition* would silently break every real inter-function call
    // compiled from source (only caught by testing a real multi-function
    // .tcc program end-to-end, not hand-built IR that names functions directly).
    std::string name = fn.name;
    if (!name.empty() && name[0] == '@') name.erase(name.begin());
    mf_.name = name;
    mf_.return_type = llt_of(fn.type ? fn.type->ret_type() : nullptr);
    mf_.has_return_value = fn.type && !fn.type->ret_type()->is_void();

    // Pre-create every MIR block up front so forward/back branch targets and
    // phi predecessor references always resolve, then reuse the CFG edges
    // the frontend's CFGPass already computed instead of recomputing them.
    for (auto& bb : fn.blocks) {
        // Qualify with the function name: block labels like "entry"/"then"
        // are only unique within one function, but multiple functions'
        // assembly can be concatenated into a single .s (see
        // CodegenDriver::lower_module_to_asm) — unqualified labels collide.
        MachineBasicBlock* mbb = mf_.add_block(name + "__" + bb->label);
        block_map_[bb.get()] = mbb;
    }
    for (auto& bb : fn.blocks) {
        MachineBasicBlock* mbb = block_map_[bb.get()];
        for (auto* pred : bb->preds) mbb->preds.push_back(block_map_[pred]);
        for (auto* succ : bb->succs) mbb->succs.push_back(block_map_[succ]);
    }

    for (auto& param : fn.params) {
        int v = vreg_for(param.get());
        mf_.param_vregs.push_back(v);
    }

    for (auto& bb : fn.blocks) {
        cur_ = block_map_[bb.get()];
        for (auto& inst : bb->insts) inst->accept(*this);
    }
}

// ── Scalar / arithmetic ──────────────────────────────────────────────────

void GenericISel::visit(ir::BinOpInst& inst) {
    int dst = vreg_for(&inst);
    LLT t = llt_of(inst.type);
    MOperand lhs = operand_for(inst.lhs.get());
    MOperand rhs = operand_for(inst.rhs.get());
    bool fp = t == LLT::F32 || t == LLT::F64;

    MOp op;
    switch (inst.op) {
        case ir::BinOpCode::Add:  op = fp ? MOp::FADD : MOp::ADD;  break;
        case ir::BinOpCode::Sub:  op = fp ? MOp::FSUB : MOp::SUB;  break;
        case ir::BinOpCode::Mul:  op = fp ? MOp::FMUL : MOp::MUL;  break;
        case ir::BinOpCode::Div:  op = fp ? MOp::FDIV : MOp::SDIV; break;
        case ir::BinOpCode::Mod:  op = MOp::SREM; break;
        case ir::BinOpCode::And:  op = MOp::AND;  break;
        case ir::BinOpCode::Or:   op = MOp::OR;   break;
        case ir::BinOpCode::Xor:  op = MOp::XOR;  break;
        case ir::BinOpCode::Shl:  op = MOp::SHL;  break;
        case ir::BinOpCode::Shr:  op = MOp::ASHR; break;
        case ir::BinOpCode::FAdd: op = MOp::FADD; break;
        case ir::BinOpCode::FSub: op = MOp::FSUB; break;
        case ir::BinOpCode::FMul: op = MOp::FMUL; break;
        case ir::BinOpCode::FDiv: op = MOp::FDIV; break;
        default: op = MOp::ADD; break;
    }
    emit(MachineInstr::make(op).def(MOperand::Reg_(dst, t)).use(lhs).use(rhs));
}

void GenericISel::visit(ir::UnOpInst& inst) {
    int dst = vreg_for(&inst);
    LLT t = llt_of(inst.type);
    MOperand src = operand_for(inst.operand.get());
    MOp op;
    switch (inst.op) {
        case ir::UnOpCode::Neg:  op = (t == LLT::F32 || t == LLT::F64) ? MOp::FNEG : MOp::NEG; break;
        case ir::UnOpCode::FNeg: op = MOp::FNEG; break;
        case ir::UnOpCode::Not:  op = MOp::NOT;  break;
        default: op = MOp::NEG; break;
    }
    emit(MachineInstr::make(op).def(MOperand::Reg_(dst, t)).use(src));
}

void GenericISel::visit(ir::CmpInst& inst) {
    int dst = vreg_for(&inst);
    MOperand lhs = operand_for(inst.lhs.get());
    MOperand rhs = operand_for(inst.rhs.get());
    bool fp = lhs.type == LLT::F32 || lhs.type == LLT::F64;

    CmpPred pred;
    switch (inst.cmp) {
        case ir::CmpCode::Eq: pred = fp ? CmpPred::FEQ : CmpPred::IEQ; break;
        case ir::CmpCode::Ne: pred = fp ? CmpPred::FNE : CmpPred::INE; break;
        case ir::CmpCode::Lt: pred = fp ? CmpPred::FLT : CmpPred::SLT; break;
        case ir::CmpCode::Le: pred = fp ? CmpPred::FLE : CmpPred::SLE; break;
        case ir::CmpCode::Gt: pred = fp ? CmpPred::FGT : CmpPred::SGT; break;
        case ir::CmpCode::Ge: pred = fp ? CmpPred::FGE : CmpPred::SGE; break;
        default: pred = CmpPred::IEQ; break;
    }
    MachineInstr mi = MachineInstr::make(fp ? MOp::FCMP : MOp::ICMP);
    mi.pred = pred;
    mi.def(MOperand::Reg_(dst, LLT::I1)).use(lhs).use(rhs);
    emit(mi);
}

// ── Memory ────────────────────────────────────────────────────────────────

void GenericISel::visit(ir::AllocaInst& inst) {
    LLT elem = llt_of(inst.alloc_type);
    int64_t size = 8;
    switch (elem) {
        case LLT::I1: case LLT::I8:  size = 1; break;
        case LLT::I16:               size = 2; break;
        case LLT::I32: case LLT::F32:size = 4; break;
        default:                     size = 8; break;
    }
    int fidx = mf_.new_frame_object(size, size < 8 ? 8 : size, /*is_spill=*/false);
    alloca_frame_index_[&inst] = fidx;
    int dst = vreg_for(&inst); // pointer to the slot
    emit(MachineInstr::make(MOp::FRAME_ADDR).def(MOperand::Reg_(dst, LLT::Ptr)).use(MOperand::Frame(fidx)));
}

void GenericISel::visit(ir::LoadInst& inst) {
    int dst = vreg_for(&inst);
    LLT t = llt_of(inst.type);
    MOperand ptr = operand_for(inst.ptr.get());
    emit(MachineInstr::make(MOp::LOAD).def(MOperand::Reg_(dst, t)).use(ptr));
}

void GenericISel::visit(ir::StoreInst& inst) {
    MOperand ptr = operand_for(inst.ptr.get());
    MOperand val = operand_for(inst.val.get());
    emit(MachineInstr::make(MOp::STORE).use(ptr).use(val));
}

// ── Control flow ──────────────────────────────────────────────────────────

void GenericISel::visit(ir::BranchInst& inst) {
    MachineBasicBlock* target = block_map_.at(inst.target);
    emit(MachineInstr::make(MOp::BR).use(MOperand::Blk(target)));
}

void GenericISel::visit(ir::CondBranchInst& inst) {
    MOperand cond = operand_for(inst.cond.get());
    MachineBasicBlock* t = block_map_.at(inst.true_bb);
    MachineBasicBlock* f = block_map_.at(inst.false_bb);
    emit(MachineInstr::make(MOp::CBR).use(cond).use(MOperand::Blk(t)).use(MOperand::Blk(f)));
}

void GenericISel::visit(ir::ReturnInst& inst) {
    MachineInstr mi = MachineInstr::make(MOp::RET);
    if (inst.val) mi.use(operand_for(inst.val->get()));
    emit(mi);
}

void GenericISel::visit(ir::CallInst& inst) {
    std::string callee = inst.callee ? inst.callee->name : "";
    if (!callee.empty() && callee[0] == '@') callee.erase(callee.begin());
    if (callee.empty()) callee = "unknown_callee";

    bool has_result = !inst.name.empty() && inst.type && !inst.type->is_void();
    MachineInstr mi = MachineInstr::make(MOp::CALL);
    mi.callee = callee;
    if (has_result) mi.def(MOperand::Reg_(vreg_for(&inst), llt_of(inst.type)));
    for (auto& a : inst.args) mi.use(operand_for(a.get()));
    emit(mi);
}

void GenericISel::visit(ir::PhiInst& inst) {
    int dst = vreg_for(&inst);
    LLT t = llt_of(inst.type);
    MachineInstr mi = MachineInstr::make(MOp::PHI);
    mi.def(MOperand::Reg_(dst, t));
    // uses are (Block, Value) pairs, one per incoming edge — consumed by
    // PhiElimination.cpp, which turns each pair into a COPY at the end of
    // the corresponding predecessor block (splitting critical edges first).
    for (auto& [val, bb] : inst.incoming) {
        MachineBasicBlock* mbb = block_map_.at(bb);
        mi.use(MOperand::Blk(mbb));
        mi.use(raw_value_operand(val.get()));
    }
    emit(mi);
}

} // namespace codegen::mir
