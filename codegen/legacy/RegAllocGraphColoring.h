#pragma once

/// codegen/legacy/RegAllocGraphColoring.h
///
/// Graph coloring register allocator using Chaitin's algorithm
///
/// Improvements over greedy allocator:
/// - Liveness analysis (not just first-to-last use)
/// - Interference graph construction
/// - Graph coloring with degree heuristics
/// - Coalescing opportunities (same-register moves eliminated)
/// - Better spilling decisions (spill lowest-cost values)
///
/// Algorithm Overview:
/// 1. Compute liveness for each variable (live-in, live-out per block)
/// 2. Build interference graph (vreg nodes, edges where live simultaneously)
/// 3. Compute degree (number of neighbors) for each node
/// 4. Iteratively remove nodes with degree < K (K = register count)
/// 5. Push removed nodes onto stack
/// 6. Color stack top-down using available colors
/// 7. If no color available, spill variable and restart

#include "MachineFunction.h"
#include <set>
#include <map>
#include <vector>
#include <bitset>
#include <memory>

namespace codegen {

struct InterferenceNode {
    int vreg;
    std::set<int> neighbors;        // vreg IDs that interfere with this one
    RegClass rclass;
    int degree() const { return neighbors.size(); }
    bool is_spilled = false;
};

struct LivenessInfo {
    std::set<int> live_in;          // vregs live at block entry
    std::set<int> live_out;         // vregs live at block exit
    std::map<int, std::set<int>> instr_live_after;  // live set after each instruction
};

class RegAllocGraphColoring {
public:
    void allocate(MachineFunction& mf);
    
    // Diagnostics
    int spill_count() const { return spill_slots_used_; }
    int regs_used_gpr() const { return gpr_used_; }
    int regs_used_fpr() const { return fpr_used_; }
    
private:
    // Constants
    static constexpr int kGPRBase    = 10;  // a0..a7
    static constexpr int kGPRCount   = 8;
    static constexpr int kFPRBase    = 10;  // fa0..fa7
    static constexpr int kFPRCount   = 8;
    static constexpr int kVecBase    = 8;   // v8..v23
    static constexpr int kVecCount   = 16;
    
    // State tracking
    int spill_slots_used_ = 0;
    int gpr_used_ = 0;
    int fpr_used_ = 0;
    
    // Phase 1: Liveness Analysis
    void compute_liveness(MachineFunction& mf);
    void build_liveness_for_block(MachineBasicBlock& bb, 
                                   std::map<int, LivenessInfo>& liveness);
    
    // Phase 2: Interference Graph
    void build_interference_graph(MachineFunction& mf);
    void add_interference(int v1, int v2, RegClass rc);
    
    // Phase 3: Graph Coloring (Chaitin's algorithm)
    void color_graph();
    void select_spill_candidate();
    void rewrite_with_allocation(MachineFunction& mf);
    
    // Helper: Determine available colors for a vreg
    std::set<int> get_available_colors(int vreg);
    
    // Data structures
    std::map<int, InterferenceNode> interference_graph_;
    std::map<int, RegClass> vreg_class_;
    std::map<int, int> vreg_allocation_;        // vreg → physical reg or -1 for spilled
    std::set<int> uncolored_vregs_;
    std::map<int, int> vreg_to_spill_slot_;     // vreg → spill slot (if spilled)
    std::map<int, LivenessInfo> block_liveness_;
};

} // namespace codegen
