#include "CallLowering.h"

namespace codegen::mir {

using target::ArgLocation;
using target::CallingConv;
using target::TargetRegisterInfo;

namespace {

MachineInstr copy_from_phys(MOperand dst_vreg, int physreg, RegClass rc) {
    return MachineInstr::make(MOp::COPY).def(dst_vreg).use(MOperand::Phys(physreg, rc, dst_vreg.type));
}
MachineInstr copy_to_phys(int physreg, RegClass rc, MOperand src) {
    return MachineInstr::make(MOp::COPY).def(MOperand::Phys(physreg, rc, src.type)).use(src);
}

void lower_entry_params(MachineFunction& mf, const CallingConv& cc) {
    // A declaration-only / empty-bodied ir::Function produces zero MIR
    // blocks (GenericISel::run() creates one block per ir::BasicBlock); with
    // nowhere to insert the parameter-binding prelude, there's nothing to do.
    if (mf.param_vregs.empty() || mf.blocks.empty()) return;
    std::vector<LLT> types;
    for (int v : mf.param_vregs) types.push_back(mf.vreg_type(v));
    std::vector<ArgLocation> locs = cc.classify_args(types);

    std::vector<MachineInstr> prelude;
    for (size_t i = 0; i < mf.param_vregs.size(); ++i) {
        LLT t = types[i];
        MOperand dst = MOperand::Reg_(mf.param_vregs[i], t);
        if (locs[i].in_register) {
            prelude.push_back(copy_from_phys(dst, locs[i].physreg, locs[i].rclass));
        } else {
            int fidx = mf.new_incoming_arg_object(8, 8);
            prelude.push_back(MachineInstr::make(MOp::LOAD).def(dst).use(MOperand::Frame(fidx)));
        }
    }
    MachineBasicBlock* entry = mf.blocks.front().get();
    entry->instrs.insert(entry->instrs.begin(), std::make_move_iterator(prelude.begin()), std::make_move_iterator(prelude.end()));
}

/// Replaces one generic CALL instruction with: arg-passing copies/stores,
/// the (now argument-free) call itself carrying the ABI's full caller-saved
/// clobber set, and a copy of the return value out of its ABI location.
std::vector<MachineInstr> lower_one_call(const MachineInstr& call, MachineFunction& mf, const CallingConv& cc, const TargetRegisterInfo& tri) {
    std::vector<LLT> arg_types;
    for (auto& u : call.uses) arg_types.push_back(u.type);
    std::vector<ArgLocation> locs = cc.classify_args(arg_types);

    std::vector<MachineInstr> out;
    for (size_t i = 0; i < call.uses.size(); ++i) {
        if (locs[i].in_register) {
            out.push_back(copy_to_phys(locs[i].physreg, locs[i].rclass, call.uses[i]));
        } else {
            int fidx = mf.new_outgoing_arg_object(8, 8);
            out.push_back(MachineInstr::make(MOp::STORE).use(MOperand::Frame(fidx)).use(call.uses[i]));
        }
    }

    MachineInstr lowered = MachineInstr::make(MOp::CALL);
    lowered.callee = call.callee;
    for (int r : tri.caller_saved(RegClass::GPR)) lowered.clobbers_gpr.push_back(r);
    for (int r : tri.caller_saved(RegClass::FPR)) lowered.clobbers_fpr.push_back(r);
    out.push_back(lowered);

    if (!call.defs.empty()) {
        LLT rt = call.defs[0].type;
        ArgLocation rloc = cc.classify_return(rt);
        out.push_back(copy_from_phys(call.defs[0], rloc.physreg, rloc.rclass));
    }
    return out;
}

void lower_calls_in_block(MachineBasicBlock& bb, MachineFunction& mf, const CallingConv& cc, const TargetRegisterInfo& tri) {
    std::vector<MachineInstr> new_instrs;
    new_instrs.reserve(bb.instrs.size());
    for (auto& mi : bb.instrs) {
        if (mi.is_generic && mi.op == MOp::CALL) {
            for (auto& lowered : lower_one_call(mi, mf, cc, tri)) new_instrs.push_back(std::move(lowered));
        } else if (mi.is_generic && mi.op == MOp::RET && !mi.uses.empty()) {
            ArgLocation rloc = cc.classify_return(mi.uses[0].type);
            new_instrs.push_back(copy_to_phys(rloc.physreg, rloc.rclass, mi.uses[0]));
            MachineInstr ret = MachineInstr::make(MOp::RET);
            ret.use(MOperand::Phys(rloc.physreg, rloc.rclass, mi.uses[0].type));
            new_instrs.push_back(std::move(ret));
        } else {
            new_instrs.push_back(mi);
        }
    }
    bb.instrs = std::move(new_instrs);
}

} // namespace

void lower_calls(MachineFunction& mf, const CallingConv& cc, const TargetRegisterInfo& tri) {
    lower_entry_params(mf, cc);
    for (auto& b : mf.blocks) lower_calls_in_block(*b, mf, cc, tri);
}

} // namespace codegen::mir
