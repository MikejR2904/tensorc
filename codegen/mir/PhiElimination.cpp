#include "PhiElimination.h"
#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

namespace codegen::mir {

namespace {

bool is_terminator(MOp op) { return op == MOp::BR || op == MOp::CBR || op == MOp::RET; }

struct PairPtrLess {
    bool operator()(const std::pair<MachineBasicBlock*, MachineBasicBlock*>& a,
                     const std::pair<MachineBasicBlock*, MachineBasicBlock*>& b) const {
        std::less<MachineBasicBlock*> lt;
        if (lt(a.first, b.first)) return true;
        if (lt(b.first, a.first)) return false;
        return lt(a.second, b.second);
    }
};
using SplitCache = std::map<std::pair<MachineBasicBlock*, MachineBasicBlock*>, MachineBasicBlock*, PairPtrLess>;

/// Returns the block where copies for the pred->succ edge belong, splitting
/// the edge with a fresh intermediate block first if it's critical (pred has
/// more than one successor AND succ has more than one predecessor).
MachineBasicBlock* edge_block(MachineFunction& mf, MachineBasicBlock* pred, MachineBasicBlock* succ, SplitCache& cache) {
    auto key = std::make_pair(pred, succ);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    bool critical = pred->succs.size() > 1 && succ->preds.size() > 1;
    if (!critical) {
        cache[key] = pred;
        return pred;
    }

    MachineBasicBlock* split = mf.add_block(pred->label + ".to." + succ->label);
    split->preds.push_back(pred);
    split->succs.push_back(succ);
    split->instrs.push_back(MachineInstr::make(MOp::BR).use(MOperand::Blk(succ)));

    if (!pred->instrs.empty()) {
        MachineInstr& term = pred->instrs.back();
        for (auto& u : term.uses)
            if (u.kind == MOperand::Block && u.block == succ) u.block = split;
    }
    for (auto& s : pred->succs) if (s == succ) s = split;
    for (auto& p : succ->preds) if (p == pred) p = split;

    cache[key] = split;
    return split;
}

} // namespace

void eliminate_phis(MachineFunction& mf) {
    SplitCache split_cache;

    // Snapshot: edge_block() may append new (phi-free) blocks to mf.blocks.
    // Iterating a separate raw-pointer snapshot keeps that from turning into
    // new "work" for this loop.
    std::vector<MachineBasicBlock*> original_blocks;
    original_blocks.reserve(mf.blocks.size());
    for (auto& b : mf.blocks) original_blocks.push_back(b.get());

    for (MachineBasicBlock* bb : original_blocks) {
        std::vector<MachineInstr*> phis;
        for (auto& mi : bb->instrs)
            if (mi.op == MOp::PHI) phis.push_back(&mi);
        if (phis.empty()) continue;

        struct PendingCopies {
            std::vector<MachineInstr> stage1; // src -> fresh temp
            std::vector<MachineInstr> stage2; // temp -> phi dst
        };
        std::unordered_map<MachineBasicBlock*, PendingCopies> pending;

        for (MachineInstr* phi : phis) {
            MOperand dst = phi->defs[0];
            for (size_t i = 0; i + 1 < phi->uses.size(); i += 2) {
                MachineBasicBlock* pred = phi->uses[i].block;
                MOperand src = phi->uses[i + 1];
                MachineBasicBlock* dest_block = edge_block(mf, pred, bb, split_cache);

                int tmp = mf.new_vreg(dst.type);
                MachineInstr stage1 =
                    (src.kind == MOperand::Imm)
                        ? MachineInstr::make(MOp::MOV_IMM).def(MOperand::Reg_(tmp, dst.type)).use(src)
                    : (src.kind == MOperand::FImm)
                        ? MachineInstr::make(MOp::FMOV_IMM).def(MOperand::Reg_(tmp, dst.type)).use(src)
                        : MachineInstr::make(MOp::COPY).def(MOperand::Reg_(tmp, dst.type)).use(src);
                MachineInstr stage2 = MachineInstr::make(MOp::COPY).def(dst).use(MOperand::Reg_(tmp, dst.type));

                pending[dest_block].stage1.push_back(std::move(stage1));
                pending[dest_block].stage2.push_back(std::move(stage2));
            }
        }

        for (auto& [block, copies] : pending) {
            auto& instrs = block->instrs;
            size_t insert_at = (!instrs.empty() && is_terminator(instrs.back().op)) ? instrs.size() - 1 : instrs.size();

            std::vector<MachineInstr> to_insert;
            to_insert.reserve(copies.stage1.size() + copies.stage2.size());
            for (auto& c : copies.stage1) to_insert.push_back(std::move(c));
            for (auto& c : copies.stage2) to_insert.push_back(std::move(c));

            instrs.insert(instrs.begin() + static_cast<std::ptrdiff_t>(insert_at),
                          std::make_move_iterator(to_insert.begin()), std::make_move_iterator(to_insert.end()));
        }

        bb->instrs.erase(std::remove_if(bb->instrs.begin(), bb->instrs.end(),
                                          [](const MachineInstr& mi) { return mi.op == MOp::PHI; }),
                          bb->instrs.end());
    }
}

} // namespace codegen::mir
