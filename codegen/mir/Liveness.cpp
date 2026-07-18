#include "Liveness.h"
#include "MIROperands.h"
#include <algorithm>
#include <climits>
#include <set>
#include <unordered_set>

namespace codegen::mir {

namespace {
using VSet = std::unordered_set<int>;
}

LivenessResult compute_liveness(const MachineFunction& mf) {
    LivenessResult result;

    // ── Number instructions in block-layout order ──────────────────────────
    std::vector<const MachineBasicBlock*> order;
    int idx = 0;
    for (auto& b : mf.blocks) {
        MachineBasicBlock* bb = b.get();
        int lo = idx;
        idx += static_cast<int>(bb->instrs.size());
        result.block_range[bb] = {lo, idx};
        order.push_back(bb);
    }

    // ── Per-block local Use/Def sets ────────────────────────────────────────
    std::unordered_map<const MachineBasicBlock*, VSet> use_set, def_set;
    std::unordered_map<int, RegClass> vreg_class;
    for (auto& b : mf.blocks) {
        MachineBasicBlock* bb = b.get();
        VSet defined_locally;
        for (auto& mi : bb->instrs) {
            for_each_use_vreg(mi, [&](const MOperand& o) {
                vreg_class[o.vreg] = o.rclass;
                if (!defined_locally.count(o.vreg)) use_set[bb].insert(o.vreg);
            });
            for_each_def_vreg(mi, [&](const MOperand& o) {
                vreg_class[o.vreg] = o.rclass;
                defined_locally.insert(o.vreg);
                def_set[bb].insert(o.vreg);
            });
        }
    }

    // ── Backward dataflow fixed point ───────────────────────────────────────
    std::unordered_map<const MachineBasicBlock*, VSet> live_in, live_out;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const MachineBasicBlock* bb = *it;
            VSet out_set;
            for (auto* succ : bb->succs) {
                auto sin = live_in.find(succ);
                if (sin != live_in.end()) out_set.insert(sin->second.begin(), sin->second.end());
            }
            VSet in_set = use_set[bb];
            const VSet& d = def_set[bb];
            for (int v : out_set) if (!d.count(v)) in_set.insert(v);

            if (out_set != live_out[bb]) { live_out[bb] = std::move(out_set); changed = true; }
            if (in_set != live_in[bb])   { live_in[bb] = std::move(in_set); changed = true; }
        }
    }

    // ── Flatten into one [start,end] interval per vreg ──────────────────────
    std::unordered_map<int, LiveInterval> acc;
    auto touch = [&](int v, RegClass rc, int pos) {
        auto it = acc.find(v);
        if (it == acc.end()) { acc[v] = LiveInterval{v, rc, pos, pos}; return; }
        it->second.start = std::min(it->second.start, pos);
        it->second.end = std::max(it->second.end, pos);
    };

    for (auto& b : mf.blocks) {
        MachineBasicBlock* bb = b.get();
        auto [lo, hi] = result.block_range[bb];
        for (int v : live_in[bb]) touch(v, vreg_class[v], lo);
        for (int v : live_out[bb]) touch(v, vreg_class[v], hi > lo ? hi - 1 : lo);

        int pos = lo;
        for (auto& mi : bb->instrs) {
            for_each_use_vreg(mi, [&](const MOperand& o) { touch(o.vreg, o.rclass, pos); });
            for_each_def_vreg(mi, [&](const MOperand& o) { touch(o.vreg, o.rclass, pos); });
            ++pos;
        }
    }

    result.intervals.reserve(acc.size());
    for (auto& [v, iv] : acc) result.intervals.push_back(iv);
    std::sort(result.intervals.begin(), result.intervals.end(),
              [](const LiveInterval& a, const LiveInterval& b) { return a.start < b.start; });
    return result;
}

} // namespace codegen::mir
