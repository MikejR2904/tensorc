/// codegen/NewCodegenDriver.cpp

#include "NewCodegenDriver.h"
#include <sstream>

namespace codegen {

ProgressiveLoweringPipeline::ProgressiveLoweringPipeline(const std::string& target)
    : target_(target), emitter_(targets::create_target_emitter(target))
{
}

bool ProgressiveLoweringPipeline::lower_tensor_op(
    ir::TensorOpInst* hlir_op,
    const std::map<const void*, TensorShape>& shapes,
    std::ostream& out)
{
    try {
        diagnostic_.clear();
        
        // ──── Phase A: HLIR → LLIR (Tiling) ────
        auto llir = tiler_.lower(hlir_op, shapes);
        if (!llir) {
            std::ostringstream oss;
            oss << "Phase A (Tiling) failed: unsupported operation code "
                << static_cast<int>(hlir_op->op);
            diagnostic_ = oss.str();
            return false;
        }
        
        out << "  ; ──── Phase A: HLIR → LLIR (Tiling) ────\n";
        out << "  ; Tiled loops for " << hlir_op->name << "\n";
        out << "  ; Loop nest depth: " << llir->dims.size() << "\n";
        out << "  ; Compute blocks: " << llir->compute_blocks.size() << "\n";
        
        // ──── Phase B: Memory Legalization & Scratchpad Allocation ────
        try {
            auto allocations = allocator_.allocate(*llir);
            out << "  ; ──── Phase B: Memory Legalization ────\n";
            out << "  ; Scratchpad allocated: " << allocator_.total_allocated() << " bytes\n";
            
            if (llir->requires_spilling) {
                out << "  ; Spilling enabled for K > 64\n";
            }
        } catch (const std::runtime_error& e) {
            diagnostic_ = std::string("Phase B (Memory allocation) failed: ") + e.what();
            return false;
        }
        
        auto legalized = legalizer_.legalize(*llir);
        if (!legalized) {
            diagnostic_ = "Phase B (Legalization) failed";
            return false;
        }
        
        // ──── Phase C: Scheduling & Double-Buffering ────
        out << "  ; ──── Phase C: Scheduling ────\n";
        auto scheduled = scheduler_.schedule(*legalized);
        
        if (scheduler_.schedule_info().uses_double_buffering) {
            out << "  ; Double-buffering enabled\n";
            out << "  ; Ping buffers: " << scheduler_.schedule_info().ping_buffers.size() << "\n";
        }
        
        // ──── Phase D: Target Emission ────
        out << "  ; ──── Phase D: Target Emission (" << target_ << ") ────\n";
        emitter_->emit(*scheduled, out);
        
        return true;
        
    } catch (const std::exception& e) {
        diagnostic_ = std::string("Lowering exception: ") + e.what();
        return false;
    }
}

} // namespace codegen
