#include "RegAlloc.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

/// codegen/RegAlloc.cpp
///
/// Two-pass greedy register allocator.
///
/// Pass 1: Scan all instructions; build a map vreg → RegClass.
///         All pre-allocated physical operands (is_vreg == false) are skipped.
///
/// Pass 2: Walk instructions again.  For each virtual register operand:
///   - If already assigned, replace with the physical register number.
///   - If unassigned and a free physical register exists, assign it.
///   - Otherwise, spill: replace with MachineOperand::spill(slot) and
///     insert a load before the instruction (and a store after if it is
///     a definition — i.e. the first operand).
///
/// The  mv a0, <vreg>  before  ret  (emitted by InstrSelector::visit(ReturnInst))
/// is handled naturally: the vreg operand gets assigned a physical register,
/// and if that register happens to already be a0 the identity mv is emitted.
/// A copy-propagation pass could remove it, but correctness is guaranteed.

namespace codegen {

static bool first_operand_is_def(const std::string& opcode)
{
    if (opcode.empty() || opcode[0] == '#') return false;
    if (opcode == "sd" || opcode == "fsd" || opcode == "j" ||
        opcode == "bnez" || opcode == "ret" || opcode == "call") {
        return false;
    }
    return true;
}

void RegAlloc::allocate(MachineFunction& mf)
{
    // ── Pass 1: collect vregs and their register classes ─────────────────
    std::unordered_map<int, RegClass> vreg_class;
    for (auto& bb : mf.blocks) {
        for (auto& mi : bb.instrs) {
            for (auto& op : mi.ops) {
                if (op.is_reg && op.is_vreg)
                    vreg_class.emplace(op.reg, op.rclass);
                if (op.is_mem && op.base_is_vreg)
                    vreg_class.emplace(op.base_reg, op.rclass);
            }
        }
    }

    // ── Allocation tables: vreg → physical reg ────────────────────────────
    std::unordered_map<int, int> gpr_assign;   // vreg → x<N>
    std::unordered_map<int, int> fpr_assign;   // vreg → f<N>
    std::unordered_map<int, int> vec_assign;   // vreg → v<N>
    std::unordered_map<int, int> spill_assign; // vreg → spill slot

    int next_gpr  = kGPRBase;
    int next_fpr  = kFPRBase;
    int next_vec  = kVecBase;
    int next_spill = 0;

    auto assign_vreg = [&](int v, RegClass rc) {
        if (gpr_assign.count(v) || fpr_assign.count(v) ||
            vec_assign.count(v) || spill_assign.count(v))
            return; // already assigned

        switch (rc) {
        case RegClass::Vector:
            if (next_vec < kVecBase + kVecCount)
                vec_assign[v] = next_vec++;
            else
                spill_assign[v] = next_spill++;
            break;
        case RegClass::FPR:
            if (next_fpr < kFPRBase + kFPRCount)
                fpr_assign[v] = next_fpr++;
            else
                spill_assign[v] = next_spill++;
            break;
        default: // GPR
            if (next_gpr < kGPRBase + kGPRCount)
                gpr_assign[v] = next_gpr++;
            else
                spill_assign[v] = next_spill++;
            break;
        }
    };

    std::vector<int> ordered_vregs;
    ordered_vregs.reserve(vreg_class.size());
    for (auto& [v, _rc] : vreg_class) ordered_vregs.push_back(v);
    std::sort(ordered_vregs.begin(), ordered_vregs.end());

    // Assign all collected vregs deterministically by virtual-register id.
    for (int v : ordered_vregs) assign_vreg(v, vreg_class[v]);
    mf.spill_slots = next_spill;

    // ── Pass 2: rewrite operands, insert spill code ───────────────────────
    for (auto& bb : mf.blocks) {
        std::vector<MachineInstr> new_instrs;
        new_instrs.reserve(bb.instrs.size() * 2);

        for (auto& mi : bb.instrs) {
            std::vector<MachineInstr> pre;  // loads before mi
            std::vector<MachineInstr> post; // stores after mi (def spills)
            bool is_def_spilled = false;

            for (size_t op_idx = 0; op_idx < mi.ops.size(); ++op_idx) {
                auto& op = mi.ops[op_idx];
                if (op.is_mem && op.base_is_vreg) {
                    int v = op.base_reg;
                    if (gpr_assign.count(v)) {
                        op.base_reg = gpr_assign[v];
                    } else if (spill_assign.count(v)) {
                        int offset = get_stack_offset(spill_assign[v]);
                        pre.push_back(MachineInstr::make("ld")
                            .add(MachineOperand::preg(5))
                            .add(MachineOperand::mem(2, offset, false)));
                        op.base_reg = 5;
                    }
                    op.base_is_vreg = false;
                    continue;
                }

                if (!op.is_reg || !op.is_vreg) continue;

                int v  = op.reg;
                bool is_def = (op_idx == 0) && first_operand_is_def(mi.opcode);

                auto assign = [&](int preg) {
                    op.reg    = preg;
                    op.is_vreg = false;
                };

                if (gpr_assign.count(v)) {
                    assign(gpr_assign[v]);
                } else if (fpr_assign.count(v)) {
                    assign(fpr_assign[v]);
                } else if (vec_assign.count(v)) {
                    assign(vec_assign[v]);
                } else if (spill_assign.count(v)) {
                    int slot   = spill_assign[v];
                    int offset = get_stack_offset(slot);
                    // Use a scratch temp: t0 (x5) for GPR, ft0 (f0) for FPR
                    int scratch = (op.rclass == RegClass::FPR) ? 0 : 5;

                    if (!is_def) {
                        // Load from spill slot into scratch before use
                        const char* ld_op = (op.rclass == RegClass::FPR) ? "fld" : "ld";
                        pre.push_back(MachineInstr::make(ld_op)
                            .add(MachineOperand::preg(scratch, op.rclass))
                            .add(MachineOperand::mem(2, offset, false)));
                    } else {
                        // After def: store scratch back to spill slot
                        const char* sd_op = (op.rclass == RegClass::FPR) ? "fsd" : "sd";
                        post.push_back(MachineInstr::make(sd_op)
                            .add(MachineOperand::preg(scratch, op.rclass))
                            .add(MachineOperand::mem(2, offset, false)));
                        is_def_spilled = true;
                    }
                    op.reg     = scratch;
                    op.is_vreg  = false;
                }
            }

            // Append: pre-loads, the instruction itself, post-stores
            for (auto& p : pre)  new_instrs.push_back(std::move(p));
            new_instrs.push_back(mi);
            for (auto& p : post) new_instrs.push_back(std::move(p));
        }

        bb.instrs = std::move(new_instrs);
    }
}

} // namespace codegen
