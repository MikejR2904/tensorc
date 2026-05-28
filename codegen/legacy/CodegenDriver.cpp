#include "InstrSelector.h"
#include "RegAlloc.h"
#include "AsmPrinter.h"
#include "MachineFunction.h"
#include "../compiler/ir/IRModule.h"
#include <fstream>

namespace codegen {

// Lower an ir::Function to assembly text file
bool lower_function_to_asm(ir::Function* fn, const std::string& out_path)
{
    MachineFunction mf; mf.name = fn->name;
    InstrSelector sel(&mf);

    for (auto& bbptr : fn->blocks) {
        mf.block_labels[bbptr.get()] = bbptr->label;
    }
    for (auto& param : fn->params) {
        mf.vreg_for(param.get());
    }

    // Walk IR and invoke visitor on instructions
    for (size_t i=0; i<fn->blocks.size(); ++i) {
        auto& bbptr = fn->blocks[i];
        sel.start_block(bbptr->label);
        for (size_t j=0; j<bbptr->insts.size(); ++j) {
            auto& instptr = bbptr->insts[j];
            instptr->accept(sel);
        }
    }

    // Regalloc
    RegAlloc ra; ra.allocate(mf);

    // Emit
    std::ofstream ofs(out_path);
    if (!ofs) return false;
    AsmPrinter printer(ofs);
    printer.print(mf);
    return true;
}

} // namespace codegen
