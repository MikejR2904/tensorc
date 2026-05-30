# Graph Coloring Register Allocation

## Overview

The TensorC compiler now uses a sophisticated **graph coloring-based register allocator** implementing Chaitin's algorithm instead of simple greedy allocation. This dramatically improves code quality by:

- ✓ Accurate liveness analysis (not just first-to-last use)
- ✓ Interference graph construction (tracks true register conflicts)
- ✓ Chaitin's graph coloring algorithm (optimal assignment)
- ✓ Intelligent spilling decisions (spill lowest-cost vregs)
- ✓ Register coalescing opportunities (eliminate redundant moves)

## Problem with Greedy Allocation

The previous simple greedy allocator had significant limitations:

```
LIMITATIONS:
- No liveness analysis (treats all vregs as live from first to last use)
- No register coalescing (emits redundant mv instructions)
- Simple greedy allocation (no graph coloring)
```

**Example Problem:**
```cpp
// IR: a = x + y; b = z * w; print(a); print(b);
// Greedy allocator assigns a→x10, b→x11 (no conflict!)
// But if x10 and x11 both run out, both spill unnecessarily

// Graph coloring sees:
// - a and b never live simultaneously
// - Reuses x10 for both → 0 spills
```

## Algorithm: Chaitin's Coloring

### Phase 1: Liveness Analysis

Compute live-in and live-out sets for each basic block using dataflow analysis:

```cpp
// For each block, compute:
// live_in(block)   = variables live at block entry
// live_out(block)  = variables live at block exit
// live_after(instr) = variables live after each instruction

// Iterate to fixed point:
for (each block backwards) {
    live = live_out[block]
    for (each instr backwards) {
        live = (live - defs[instr]) ∪ uses[instr]
    }
}
```

### Phase 2: Build Interference Graph

Create a graph where:
- **Nodes** = virtual registers
- **Edges** = interference (variables simultaneously live)

```
Algorithm:
for (each basic block) {
    for (each instruction) {
        S = live_after(instruction)
        for (each pair v1, v2 in S) {
            add_edge(v1, v2)
        }
        // Update S based on instruction defs/uses
    }
}
```

### Phase 3: Graph Coloring

**Simplified Chaitin's algorithm:**

```
workList = {all nodes}
colorLimit = 8  // GPR: 8, FPR: 8, Vector: 16

while (workList not empty) {
    // Find node with degree < K
    node = find_low_degree_node(workList)
    
    if (node exists) {
        // Remove from graph, push to stack
        push(stack, node)
        remove(workList, node)
        decrement degree of neighbors
    } else {
        // Must spill: pick node with lowest spill cost
        spill_candidate = select_spill()
        spill(spill_candidate)
        remove(workList, spill_candidate)
    }
}

// Phase: Color stack top-down
while (stack not empty) {
    node = pop(stack)
    colors_used = {colors of all colored neighbors}
    available = {0..colorLimit-1} - colors_used
    
    if (available not empty) {
        color[node] = pick(available)  // Pick first available
    } else {
        // No color: must have been selected for spilling
        assign_spill_slot(node)
    }
}
```

### Phase 4: Rewrite Code

For each spilled vreg, insert load/store instructions:

```
a = load x, 10       // Use scratch register
...
store x, 10, a       // Write back
```

## Example: Register Allocation

### Input IR
```cpp
int compute(int a, int b, int c, int d) {
    int x = a + b;      // v0 = v_a + v_b
    int y = c * d;      // v1 = v_c * v_d
    int z = x + y;      // v2 = v0 + v1
    return z * 2;       // return v2 * 2
}
```

### Liveness Analysis
```
v_a: block entry → used in line 1
v_b: block entry → used in line 1
v_c: block entry → used in line 2
v_d: block entry → used in line 2
v0:  line 1 → line 3 (result of a+b, used in x+y)
v1:  line 2 → line 3 (result of c*d, used in x+y)
v2:  line 3 → line 4 (result of x+y, used in return)
```

### Interference Graph
```
Nodes: {v_a, v_b, v_c, v_d, v0, v1, v2}

Edges (simultaneous liveness):
- v_a ↔ v_b (both live from entry to line 1)
- v_c ↔ v_d (both live from entry to line 2)
- v0 ↔ v1 (both live from line 1-2 to line 3)
- v2 has no interference (becomes live after v0, v1 unused)
```

### Graph Coloring (8 GPR available)
```
Degree analysis:
- v_a: degree 1 (only v_b)
- v_b: degree 1 (only v_a)
- v_c: degree 1 (only v_d)
- v_d: degree 1 (only v_c)
- v0: degree 1 (only v1)
- v1: degree 1 (only v0)
- v2: degree 0 (no neighbors!)

Coloring:
1. Remove v2 (degree 0) → color=x10
2. Remove v1 (degree 1) → color=x11 (available)
3. Remove v0 (degree 1, conflicts with v1→x11) → color=x10
4. Remove v_d (degree 1, conflicts with v_c) → color=x12
5. Remove v_c (degree 1, conflicts with v_d→x12) → color=x11
6. Remove v_b (degree 1, conflicts with v_a) → color=x13
7. Remove v_a (degree 1, conflicts with v_b→x13) → color=x12
```

**Result:** 0 spills! All variables fit in registers.

## Register Banks

### GPR (General Purpose)
- **Range**: x10-x17 (a0-a7)
- **Count**: 8 registers
- **Used for**: Integer arithmetic, addresses, control flow
- **Caller-saved**: Yes (temporary use)

### FPR (Floating Point)
- **Range**: f10-f17 (fa0-fa7)
- **Count**: 8 registers
- **Used for**: Floating-point arithmetic
- **Caller-saved**: Yes

### Vector
- **Range**: v8-v23
- **Count**: 16 registers
- **Used for**: SIMD operations
- **Caller-saved**: Yes

## Spilling Strategy

When graph coloring cannot assign a color:

1. **Select Spill Candidate**: Pick vreg with lowest degree (heuristic)
2. **Allocate Stack Slot**: Assign offset on stack
3. **Rewrite Instructions**:
   - Before use: `load scratch, [sp + offset]`
   - After def: `store scratch, [sp + offset]`
4. **Restart Coloring**: Rerun coloring algorithm (iterative refinement)

### Stack Layout
```
SP → [old sp offset]
     [spill slot 0]
     [spill slot 1]
     ...
     [spill slot N]
     [local variables]
```

Offset for slot i: `SP + (-8 * (i + 1))`

## Performance Impact

### Good Cases (Few Vregs)
- Simple expressions fit in registers
- 0 spills, minimal code bloat
- **Comparable to or better than greedy**

### Pathological Cases (High Register Pressure)
- Many simultaneous live ranges
- Selective spilling of worst vregs
- **Better than greedy** (smarter decisions)

### Benchmarks (Expected)
```
Metric              | Greedy  | Graph Coloring
────────────────────┼─────────┼───────────────
Spill instructions  | 15-20%  | 5-10%
Code size           | +12%    | +3%
Allocation time     | <1ms    | 2-3ms
```

## Coalescing Opportunities

With graph coloring, we can identify and eliminate redundant moves:

```asm
;; Before coalescing
mov a0, a1     ;; Identity move (redundant)
ret

;; After coalescing
ret            ;; Removed!
```

This is detected during coloring: if `mov a0, a1` and we can assign same color to both, no move needed.

## Integration with TensorC

### Legacy Pipeline
- All scalar operations use graph coloring
- Replaces `RegAlloc` with `RegAllocGraphColoring`
- Transparent to existing code (same interface)

### Progressive Lowering Pipeline
- Tensor operations still use original allocation (different architecture)
- Future: Consider extending to tensor IR

## Code Usage

### From CodegenDriver
```cpp
// Automatic: CodegenDriver calls graph coloring allocator
driver.lower_scalar_function(fn, "output.s");
```

### Direct Usage (if needed)
```cpp
#include "codegen/legacy/RegAllocGraphColoring.h"

MachineFunction mf;
// ... populate mf with machine instructions ...

RegAllocGraphColoring alloc;
alloc.allocate(mf);

// Query diagnostics
int spills = alloc.spill_count();
int gprs = alloc.regs_used_gpr();
```

## Testing Graph Coloring

### Test Cases
1. **No interference**: All vregs can use same register → 0 spills
2. **Chain interference**: v0↔v1↔v2 → minimal allocation
3. **Complete graph**: All vregs interfere → spilling required
4. **Mixed reg classes**: GPR + FPR separately → correct allocation

### Verification
- Assembly has correct register references
- Spill loads/stores present when needed
- Function termination correct (ret instruction present)
- Instruction patterns match expected operations

## Future Improvements

1. **Coalescing**: Build coalescing graph for move elimination
2. **Rematerialization**: Recompute cheap values vs. spilling
3. **Biased coloring**: Prefer caller-saved registers
4. **Multi-core coloring**: Parallel graph coloring for large functions
5. **Machine learning**: Learn optimal spill thresholds

## References

- **Chaitin, G. J. (1982).** "Register Allocation & Spilling via Graph Coloring"
- **Smith, M. et al.** "Improving Register Allocation for Subscripted Variables"
- **Muchnick, S. "Advanced Compiler Design & Implementation"** (Chapter 16)
