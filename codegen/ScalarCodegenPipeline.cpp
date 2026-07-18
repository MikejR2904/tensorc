#include "ScalarCodegenPipeline.h"
#include "mir/GenericISel.h"
#include "mir/PhiElimination.h"
#include "mir/CallLowering.h"
#include "mir/LinearScanRegAlloc.h"
#include "mir/PrologueEpilogInserter.h"

namespace codegen {

ScalarCodegenPipeline::ScalarCodegenPipeline(const std::string& target_name)
    : target_(target::create_target(target_name)) {}

bool ScalarCodegenPipeline::lower_function(ir::Function& fn, std::ostream& out) {
    if (!target_) return false;

    mir::MachineFunction mf;
    mir::GenericISel isel(mf);
    isel.run(fn);

    mir::eliminate_phis(mf);
    mir::lower_calls(mf, target_->calling_conv(), target_->reg_info());

    for (auto& b : mf.blocks) {
        std::vector<mir::MachineInstr> selected;
        selected.reserve(b->instrs.size());
        for (auto& mi : b->instrs) {
            if (mi.is_generic) {
                for (auto& lowered : target_->instr_info().select(mi, mf)) selected.push_back(std::move(lowered));
            } else {
                selected.push_back(mi);
            }
        }
        b->instrs = std::move(selected);
    }

    mir::LinearScanRegAlloc regalloc;
    regalloc.allocate(mf, target_->reg_info(), target_->instr_info());

    mir::insert_prologue_epilogue(mf, target_->reg_info(), target_->instr_info(), target_->frame_lowering());

    target_->asm_printer().print(out, mf, target_->instr_info(), target_->reg_info());
    return true;
}

} // namespace codegen
