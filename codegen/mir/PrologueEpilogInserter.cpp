#include "PrologueEpilogInserter.h"

namespace codegen::mir {

using target::FrameLayout;
using target::FrameLowering;
using target::SavedReg;
using target::TargetInstrInfo;
using target::TargetRegisterInfo;

namespace {

void resolve_operand(MOperand& o, const MachineFunction& mf, int fp_physreg) {
    if (o.kind == MOperand::Mem && o.frame_idx >= 0) {
        int64_t final_off = mf.frame_objects[static_cast<size_t>(o.frame_idx)].offset + o.imm;
        o.frame_idx = -1;
        o.vreg = fp_physreg;
        o.is_physical = true;
        o.imm = final_off;
    } else if (o.kind == MOperand::FrameIdx) {
        int64_t final_off = mf.frame_objects[static_cast<size_t>(o.frame_idx)].offset;
        o.kind = MOperand::Imm;
        o.imm = final_off;
    }
}

} // namespace

void insert_prologue_epilogue(MachineFunction& mf, const TargetRegisterInfo& tri, const TargetInstrInfo& tii, FrameLowering& fl) {
    // A declaration-only / empty-bodied function has no blocks to frame —
    // nothing to lay out and nowhere to splice a prologue/epilogue into.
    if (mf.blocks.empty()) return;

    // ── Layout: negative-growing region below FP (ra save, callee-saved
    //    regs, spills/locals, outgoing-call-arg space nearest SP), and a
    //    separate positive region above FP for incoming stack arguments. ──
    int64_t offset = 0;
    bool save_ra = tri.ra() >= 0;
    int64_t ra_offset = 0;
    if (save_ra) { offset -= 8; ra_offset = offset; }

    std::vector<SavedReg> saved;
    for (int r : mf.used_callee_saved_gpr) { offset -= 8; saved.push_back({r, RegClass::GPR, offset}); }
    for (int r : mf.used_callee_saved_fpr) { offset -= 8; saved.push_back({r, RegClass::FPR, offset}); }

    for (auto& fo : mf.frame_objects) {
        if (fo.is_incoming_arg || fo.is_outgoing_arg) continue;
        offset -= 8;
        fo.offset = offset;
        fo.resolved = true;
    }
    for (auto& fo : mf.frame_objects) {
        if (!fo.is_outgoing_arg) continue;
        offset -= 8;
        fo.offset = offset;
        fo.resolved = true;
    }

    int64_t frame_size = -offset;
    int align = tri.stack_alignment();
    frame_size = (frame_size + align - 1) / align * align;

    int64_t in_off = tri.incoming_args_base_offset();
    for (auto& fo : mf.frame_objects) {
        if (!fo.is_incoming_arg) continue;
        fo.offset = in_off;
        fo.resolved = true;
        in_off += 8;
    }

    // ── Resolve every Mem/FrameIdx operand across the whole function ───────
    int fp = tri.fp();
    for (auto& b : mf.blocks) {
        for (auto& mi : b->instrs) {
            for (auto& o : mi.defs) resolve_operand(o, mf, fp);
            for (auto& o : mi.uses) resolve_operand(o, mf, fp);
            for (auto& o : mi.print_operands) resolve_operand(o, mf, fp);
        }
    }

    // ── Splice in the real prologue/epilogue ────────────────────────────────
    FrameLayout layout;
    layout.frame_size = frame_size;
    layout.save_ra = save_ra;
    layout.ra_offset = ra_offset;
    layout.saved_callee_regs = saved;

    MachineBasicBlock* entry = mf.blocks.front().get();
    auto prologue = fl.emit_prologue(layout);
    entry->instrs.insert(entry->instrs.begin(), std::make_move_iterator(prologue.begin()), std::make_move_iterator(prologue.end()));

    for (auto& b : mf.blocks) {
        std::vector<MachineInstr> new_instrs;
        new_instrs.reserve(b->instrs.size() + 4);
        for (auto& mi : b->instrs) {
            if (!mi.is_generic && tii.describe(mi.target_op).is_return) {
                for (auto& e : fl.emit_epilogue(layout)) new_instrs.push_back(std::move(e));
            }
            new_instrs.push_back(mi);
        }
        b->instrs = std::move(new_instrs);
    }
}

} // namespace codegen::mir
