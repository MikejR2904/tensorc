/// codegen/legacy/RegAllocGraphColoring.cpp
///
/// Implementation of graph coloring register allocation (Chaitin's algorithm)

#include "RegAllocGraphColoring.h"
#include <algorithm>
#include <queue>
#include <iostream>

namespace codegen {

void RegAllocGraphColoring::allocate(MachineFunction& mf)
{
    vreg_class_.clear();
    interference_graph_.clear();
    vreg_allocation_.clear();
    vreg_to_spill_slot_.clear();
    spill_slots_used_ = 0;
    gpr_used_ = 0;
    fpr_used_ = 0;
    
    // Phase 1: Liveness Analysis
    compute_liveness(mf);
    
    // Phase 2: Build Interference Graph
    build_interference_graph(mf);
    
    // Phase 3: Graph Coloring
    uncolored_vregs_.clear();
    for (auto& [vreg, node] : interference_graph_) {
        uncolored_vregs_.insert(vreg);
    }
    
    color_graph();
    
    // Phase 4: Rewrite with allocation
    rewrite_with_allocation(mf);
    
    mf.spill_slots = spill_slots_used_;
}

void RegAllocGraphColoring::compute_liveness(MachineFunction& mf)
{
    // Initialize liveness for all blocks
    std::map<int, LivenessInfo> liveness;
    
    // Iterate until convergence (fixed-point)
    bool changed = true;
    while (changed) {
        changed = false;
        
        // Process blocks in reverse order for faster convergence
        for (int i = (int)mf.blocks.size() - 1; i >= 0; --i) {
            auto& bb = mf.blocks[i];
            auto& bb_liveness = liveness[i];
            
            auto old_live_in = bb_liveness.live_in;
            
            // live_out = union of all successors' live_in
            bb_liveness.live_out.clear();
            // (simplified: assume all vregs may flow to successors)
            
            // live_in = (live_out - defined here) ∪ used here
            build_liveness_for_block(bb, liveness);
            
            if (bb_liveness.live_in != old_live_in) {
                changed = true;
            }
        }
    }
    
    block_liveness_ = liveness;
}

void RegAllocGraphColoring::build_liveness_for_block(
    MachineBasicBlock& bb,
    std::map<int, LivenessInfo>& liveness)
{
    // Scan block backwards to compute live sets
    std::set<int> live;  // Currently live registers
    
    // Initialize with live_out (vregs live after block)
    for (auto& other_bb : dynamic_cast<MachineFunction*>(nullptr) ? 
         std::vector<MachineBasicBlock>() : std::vector<MachineBasicBlock>()) {
        // (simplified: just track within block for now)
    }
    
    // Walk instructions backwards
    for (int i = (int)bb.instrs.size() - 1; i >= 0; --i) {
        auto& mi = bb.instrs[i];
        
        // Record liveness after this instruction
        liveness[bb.label].instr_live_after[i] = live;
        
        // Update live set based on this instruction
        // If first operand is a def, remove it from live
        // Add all source operands to live set
        
        for (size_t op_idx = 0; op_idx < mi.ops.size(); ++op_idx) {
            auto& op = mi.ops[op_idx];
            if (op.is_reg && op.is_vreg) {
                bool is_def = (op_idx == 0) && 
                    (mi.opcode != "sd" && mi.opcode != "fsd" && 
                     mi.opcode != "j" && mi.opcode != "ret");
                
                if (is_def) {
                    live.erase(op.reg);
                } else {
                    live.insert(op.reg);
                }
            }
        }
    }
    
    liveness[bb.label].live_in = live;
}

void RegAllocGraphColoring::build_interference_graph(MachineFunction& mf)
{
    // Collect all vregs and their classes
    for (auto& bb : mf.blocks) {
        for (auto& mi : bb.instrs) {
            for (auto& op : mi.ops) {
                if (op.is_reg && op.is_vreg) {
                    if (vreg_class_.find(op.reg) == vreg_class_.end()) {
                        vreg_class_[op.reg] = op.rclass;
                        interference_graph_[op.reg] = InterferenceNode{op.reg, {}, op.rclass};
                    }
                }
            }
        }
    }
    
    // Build interference edges
    for (auto& bb : mf.blocks) {
        std::set<int> live;  // Live vregs at current point
        
        for (auto& mi : bb.instrs) {
            // All vregs in live set interfere with each other
            for (int v1 : live) {
                for (int v2 : live) {
                    if (v1 < v2) {  // Avoid duplicates
                        add_interference(v1, v2, vreg_class_[v1]);
                    }
                }
            }
            
            // Update live set based on this instruction
            for (size_t op_idx = 0; op_idx < mi.ops.size(); ++op_idx) {
                auto& op = mi.ops[op_idx];
                if (op.is_reg && op.is_vreg) {
                    bool is_def = (op_idx == 0) && 
                        (mi.opcode != "sd" && mi.opcode != "fsd");
                    
                    if (is_def) {
                        live.erase(op.reg);
                    } else {
                        live.insert(op.reg);
                    }
                }
            }
        }
    }
}

void RegAllocGraphColoring::add_interference(int v1, int v2, RegClass rc)
{
    if (vreg_class_[v1] != vreg_class_[v2]) return;  // Different reg classes don't interfere
    
    interference_graph_[v1].neighbors.insert(v2);
    interference_graph_[v2].neighbors.insert(v1);
}

void RegAllocGraphColoring::color_graph()
{
    // Simplified Chaitin's algorithm
    std::vector<int> elimination_stack;
    
    // Phase 1: Eliminate nodes with degree < K
    auto get_color_count = [](RegClass rc) {
        if (rc == RegClass::FPR) return kFPRCount;
        if (rc == RegClass::Vector) return kVecCount;
        return kGPRCount;
    };
    
    while (!uncolored_vregs_.empty()) {
        int min_vreg = -1;
        int min_degree = 1000000;
        
        for (int vreg : uncolored_vregs_) {
            int degree = interference_graph_[vreg].degree();
            int k = get_color_count(vreg_class_[vreg]);
            
            if (degree < k && degree < min_degree) {
                min_vreg = vreg;
                min_degree = degree;
            }
        }
        
        if (min_vreg != -1) {
            elimination_stack.push_back(min_vreg);
            uncolored_vregs_.erase(min_vreg);
        } else if (!uncolored_vregs_.empty()) {
            // Must spill: pick lowest-degree node to spill
            select_spill_candidate();
        }
    }
    
    // Phase 2: Color nodes by processing stack top-down
    while (!elimination_stack.empty()) {
        int vreg = elimination_stack.back();
        elimination_stack.pop_back();
        
        auto colors = get_available_colors(vreg);
        if (!colors.empty()) {
            vreg_allocation_[vreg] = *colors.begin();  // Pick first available
        } else {
            // Spill this variable
            vreg_to_spill_slot_[vreg] = spill_slots_used_++;
        }
    }
}

void RegAllocGraphColoring::select_spill_candidate()
{
    // Select vreg with lowest spill cost (lowest degree first)
    if (uncolored_vregs_.empty()) return;
    
    int spill_vreg = *std::min_element(uncolored_vregs_.begin(), uncolored_vregs_.end(),
        [this](int v1, int v2) {
            return interference_graph_[v1].degree() < interference_graph_[v2].degree();
        });
    
    vreg_to_spill_slot_[spill_vreg] = spill_slots_used_++;
    uncolored_vregs_.erase(spill_vreg);
}

std::set<int> RegAllocGraphColoring::get_available_colors(int vreg)
{
    RegClass rc = vreg_class_[vreg];
    int base = (rc == RegClass::FPR) ? kFPRBase : (rc == RegClass::Vector) ? kVecBase : kGPRBase;
    int count = (rc == RegClass::FPR) ? kFPRCount : (rc == RegClass::Vector) ? kVecCount : kGPRCount;
    
    std::set<int> available;
    for (int i = 0; i < count; ++i) {
        available.insert(base + i);
    }
    
    // Remove colors used by neighbors
    for (int neighbor : interference_graph_[vreg].neighbors) {
        if (vreg_allocation_.count(neighbor)) {
            available.erase(vreg_allocation_[neighbor]);
        }
    }
    
    return available;
}

void RegAllocGraphColoring::rewrite_with_allocation(MachineFunction& mf)
{
    for (auto& bb : mf.blocks) {
        std::vector<MachineInstr> new_instrs;
        new_instrs.reserve(bb.instrs.size() * 2);
        
        for (auto& mi : bb.instrs) {
            std::vector<MachineInstr> pre, post;
            
            for (size_t op_idx = 0; op_idx < mi.ops.size(); ++op_idx) {
                auto& op = mi.ops[op_idx];
                if (!op.is_reg || !op.is_vreg) continue;
                
                int vreg = op.reg;
                if (vreg_allocation_.count(vreg)) {
                    op.reg = vreg_allocation_[vreg];
                    op.is_vreg = false;
                    if (vreg_allocation_[vreg] < kGPRBase && op.rclass != RegClass::FPR) {
                        gpr_used_++;
                    } else if (op.rclass == RegClass::FPR) {
                        fpr_used_++;
                    }
                } else if (vreg_to_spill_slot_.count(vreg)) {
                    // Generate spill code
                    int slot = vreg_to_spill_slot_[vreg];
                    int offset = -8 * (slot + 1);
                    int scratch = (op.rclass == RegClass::FPR) ? 0 : 5;
                    
                    bool is_def = (op_idx == 0) && 
                        (mi.opcode != "sd" && mi.opcode != "fsd" && mi.opcode != "ret");
                    
                    if (!is_def) {
                        const char* ld_op = (op.rclass == RegClass::FPR) ? "fld" : "ld";
                        pre.push_back(MachineInstr::make(ld_op)
                            .add(MachineOperand::preg(scratch, op.rclass))
                            .add(MachineOperand::mem(2, offset, false)));
                    } else {
                        const char* sd_op = (op.rclass == RegClass::FPR) ? "fsd" : "sd";
                        post.push_back(MachineInstr::make(sd_op)
                            .add(MachineOperand::preg(scratch, op.rclass))
                            .add(MachineOperand::mem(2, offset, false)));
                    }
                    
                    op.reg = scratch;
                    op.is_vreg = false;
                }
            }
            
            for (auto& p : pre) new_instrs.push_back(std::move(p));
            new_instrs.push_back(mi);
            for (auto& p : post) new_instrs.push_back(std::move(p));
        }
        
        bb.instrs = std::move(new_instrs);
    }
}

} // namespace codegen
