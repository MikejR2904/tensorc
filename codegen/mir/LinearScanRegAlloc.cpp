#include "LinearScanRegAlloc.h"
#include "Liveness.h"
#include "MIROperands.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace codegen::mir {

using target::TargetInstrInfo;
using target::TargetRegisterInfo;

namespace {

struct BusyTable {
    // [class][physreg] -> sorted instruction positions where that physreg is
    // pinned by an explicit Phys operand or a CALL's clobber list.
    std::unordered_map<int, std::vector<int>> table[2];

    bool busy_during(RegClass rc, int physreg, int start, int end) const {
        auto it = table[static_cast<int>(rc)].find(physreg);
        if (it == table[static_cast<int>(rc)].end()) return false;
        auto lo = std::lower_bound(it->second.begin(), it->second.end(), start);
        return lo != it->second.end() && *lo <= end;
    }
};

BusyTable build_busy_table(const MachineFunction& mf, const std::unordered_map<const MachineInstr*, int>& pos_of) {
    BusyTable bt;
    for (auto& b : mf.blocks) {
        for (auto& mi : b->instrs) {
            int p = pos_of.at(&mi);
            auto scan = [&](const std::vector<MOperand>& ops) {
                for (auto& o : ops) if (o.kind == MOperand::Reg && o.is_physical) bt.table[static_cast<int>(o.rclass)][o.vreg].push_back(p);
            };
            scan(mi.defs);
            scan(mi.uses);
            for (int r : mi.clobbers_gpr) bt.table[static_cast<int>(RegClass::GPR)][r].push_back(p);
            for (int r : mi.clobbers_fpr) bt.table[static_cast<int>(RegClass::FPR)][r].push_back(p);
        }
    }
    for (auto& cls : bt.table) for (auto& [_, v] : cls) std::sort(v.begin(), v.end());
    return bt;
}

struct ActiveEntry { LiveInterval iv; int physreg; };

/// One class's worth of linear scan. `pool` excludes the two scratch
/// registers reserved for spill code (see allocate()).
struct ClassAllocResult {
    std::unordered_map<int, int> assigned; // vreg -> physreg
    std::unordered_set<int> spilled;       // vreg ids that didn't fit
};

ClassAllocResult linear_scan_one_class(const std::vector<LiveInterval>& intervals, const std::vector<int>& pool, const BusyTable& busy, RegClass rc) {
    ClassAllocResult result;
    std::vector<ActiveEntry> active; // kept sorted by iv.end ascending

    auto pick_free = [&](int start, int end, const std::unordered_set<int>& held) -> int {
        for (int r : pool) {
            if (held.count(r)) continue;
            if (busy.busy_during(rc, r, start, end)) continue;
            return r;
        }
        return -1;
    };

    for (const auto& iv : intervals) {
        if (iv.rclass != rc) continue;

        // Expire active intervals that ended before this one starts.
        active.erase(std::remove_if(active.begin(), active.end(),
                                     [&](const ActiveEntry& e) { return e.iv.end < iv.start; }),
                     active.end());

        std::unordered_set<int> held;
        for (auto& e : active) held.insert(e.physreg);

        int reg = pick_free(iv.start, iv.end, held);
        if (reg != -1) {
            result.assigned[iv.vreg] = reg;
            active.push_back({iv, reg});
            std::sort(active.begin(), active.end(), [](const ActiveEntry& a, const ActiveEntry& b) { return a.iv.end < b.iv.end; });
            continue;
        }

        // No free register: evict the active interval with the furthest end
        // if that helps (classic Poletto-Sarkar SpillAtInterval); otherwise
        // spill the current interval itself.
        if (!active.empty() && active.back().iv.end > iv.end) {
            ActiveEntry evicted = active.back();
            active.pop_back();
            result.spilled.insert(evicted.iv.vreg);
            result.assigned.erase(evicted.iv.vreg);

            held.erase(evicted.physreg);
            int retry = pick_free(iv.start, iv.end, held);
            if (retry != -1) {
                result.assigned[iv.vreg] = retry;
                active.push_back({iv, retry});
                std::sort(active.begin(), active.end(), [](const ActiveEntry& a, const ActiveEntry& b) { return a.iv.end < b.iv.end; });
                continue;
            }
        }
        result.spilled.insert(iv.vreg);
    }
    return result;
}

} // namespace

void LinearScanRegAlloc::allocate(MachineFunction& mf, const TargetRegisterInfo& tri, const TargetInstrInfo& tii) {
    // ── Phase A (read-only): liveness + physreg-busy table on the
    //    unmodified instruction stream ──────────────────────────────────────
    LivenessResult live = compute_liveness(mf);

    std::unordered_map<const MachineInstr*, int> pos_of;
    { int p = 0; for (auto& b : mf.blocks) for (auto& mi : b->instrs) pos_of[&mi] = p++; }
    BusyTable busy = build_busy_table(mf, pos_of);

    auto pool_for = [&](RegClass rc) {
        std::vector<int> pool = tri.allocatable(rc);
        // Reserve the last two entries as spill-code scratch registers so a
        // spilled operand can always be reloaded/stored without contending
        // with the main allocation (worst case: two distinct spilled vregs —
        // e.g. an x86 RMW's dst and its rhs — in the same instruction).
        if (pool.size() > 2) pool.resize(pool.size() - 2);
        return pool;
    };
    std::vector<int> gpr_pool = pool_for(RegClass::GPR);
    std::vector<int> fpr_pool = pool_for(RegClass::FPR);
    std::vector<int> full_gpr = tri.allocatable(RegClass::GPR);
    std::vector<int> full_fpr = tri.allocatable(RegClass::FPR);
    int gpr_scratch[2] = {full_gpr[full_gpr.size() - 2], full_gpr[full_gpr.size() - 1]};
    int fpr_scratch[2] = {full_fpr[full_fpr.size() - 2], full_fpr[full_fpr.size() - 1]};

    ClassAllocResult gpr_res = linear_scan_one_class(live.intervals, gpr_pool, busy, RegClass::GPR);
    ClassAllocResult fpr_res = linear_scan_one_class(live.intervals, fpr_pool, busy, RegClass::FPR);

    std::unordered_map<int, int> assigned = gpr_res.assigned;
    assigned.insert(fpr_res.assigned.begin(), fpr_res.assigned.end());

    // ── Record callee-saved usage for PrologueEpilogInserter ───────────────
    std::unordered_set<int> cs_gpr(tri.callee_saved(RegClass::GPR).begin(), tri.callee_saved(RegClass::GPR).end());
    std::unordered_set<int> cs_fpr(tri.callee_saved(RegClass::FPR).begin(), tri.callee_saved(RegClass::FPR).end());
    std::unordered_set<int> used_cs_gpr, used_cs_fpr;
    for (auto& [v, r] : gpr_res.assigned) if (cs_gpr.count(r)) used_cs_gpr.insert(r);
    for (auto& [v, r] : fpr_res.assigned) if (cs_fpr.count(r)) used_cs_fpr.insert(r);
    mf.used_callee_saved_gpr.assign(used_cs_gpr.begin(), used_cs_gpr.end());
    mf.used_callee_saved_fpr.assign(used_cs_fpr.begin(), used_cs_fpr.end());
    std::sort(mf.used_callee_saved_gpr.begin(), mf.used_callee_saved_gpr.end());
    std::sort(mf.used_callee_saved_fpr.begin(), mf.used_callee_saved_fpr.end());

    // ── Phase B: rewrite operands + insert spill code ──────────────────────
    std::unordered_map<int, int> spill_slot; // vreg -> frame_object index
    auto slot_for = [&](int vreg) {
        auto it = spill_slot.find(vreg);
        if (it != spill_slot.end()) return it->second;
        int idx = mf.new_frame_object(8, 8, /*is_spill=*/true);
        spill_slot[vreg] = idx;
        return idx;
    };
    std::unordered_map<int, RegClass> vreg_rc; // rclass is stable per vreg across the whole function
    for (auto& iv : live.intervals) vreg_rc[iv.vreg] = iv.rclass;

    for (auto& b : mf.blocks) {
        std::vector<MachineInstr> new_instrs;
        new_instrs.reserve(b->instrs.size());

        for (auto& mi : b->instrs) {
            std::unordered_map<int, int> scratch_this_instr; // spilled vreg -> scratch physreg
            std::unordered_set<int> need_load, need_store;
            int gpr_idx = 0, fpr_idx = 0;

            auto ensure_scratch = [&](const MOperand& o) {
                if (assigned.count(o.vreg) || scratch_this_instr.count(o.vreg)) return;
                int s = (o.rclass == RegClass::GPR) ? gpr_scratch[gpr_idx++] : fpr_scratch[fpr_idx++];
                scratch_this_instr[o.vreg] = s;
            };
            for_each_def_vreg(mi, [&](const MOperand& o) { if (!assigned.count(o.vreg)) { ensure_scratch(o); need_store.insert(o.vreg); } });
            for_each_use_vreg(mi, [&](const MOperand& o) { if (!assigned.count(o.vreg)) { ensure_scratch(o); need_load.insert(o.vreg); } });

            std::vector<MachineInstr> pre, post;
            for (auto& [vreg, scratch] : scratch_this_instr) {
                RegClass rc = vreg_rc.count(vreg) ? vreg_rc[vreg] : RegClass::GPR;
                int fidx = slot_for(vreg);
                if (need_load.count(vreg)) {
                    unsigned op = tii.spill_load_opcode(rc);
                    pre.push_back(MachineInstr::makeTarget(op).def(MOperand::Phys(scratch, rc, LLT::I64)).use(MOperand::MemFrame(fidx, LLT::I64))
                                      .pr(MOperand::MemFrame(fidx, LLT::I64)).pr(MOperand::Phys(scratch, rc, LLT::I64)));
                }
            }

            // Rewrite this instruction's operands (defs, uses, print_operands).
            for_each_vreg_operand(mi, [&](MOperand& o) {
                if (assigned.count(o.vreg)) { o.vreg = assigned[o.vreg]; o.is_physical = true; }
                else { o.vreg = scratch_this_instr[o.vreg]; o.is_physical = true; }
            });

            for (auto& [vreg, scratch] : scratch_this_instr) {
                if (need_store.count(vreg)) {
                    RegClass rc = vreg_rc.count(vreg) ? vreg_rc[vreg] : RegClass::GPR;
                    int fidx = slot_for(vreg);
                    unsigned op = tii.spill_store_opcode(rc);
                    post.push_back(MachineInstr::makeTarget(op).use(MOperand::MemFrame(fidx, LLT::I64)).use(MOperand::Phys(scratch, rc, LLT::I64))
                                       .pr(MOperand::Phys(scratch, rc, LLT::I64)).pr(MOperand::MemFrame(fidx, LLT::I64)));
                }
            }

            for (auto& p : pre) new_instrs.push_back(std::move(p));
            new_instrs.push_back(mi);
            for (auto& p : post) new_instrs.push_back(std::move(p));
        }

        b->instrs = std::move(new_instrs);
    }
}

} // namespace codegen::mir
