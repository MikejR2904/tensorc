# Register Allocation

## Overview

TensorC's register allocator is a **linear-scan allocator** (Poletto-Sarkar
algorithm) driven by real liveness analysis, implemented in
`codegen/mir/Liveness.h/.cpp` and `codegen/mir/LinearScanRegAlloc.h/.cpp`.
It replaced two earlier implementations:

- A greedy allocator (`codegen/legacy/RegAlloc.cpp`, now deleted) that
  assigned virtual registers to physical ones in ID order with **no
  liveness analysis** — every vreg was treated as live for the whole
  function, so it spilled far more than necessary (visible in the old
  `test_legacy_reg_pressure.s` sample output, which spilled three
  temporaries in a function small enough that none should have been
  needed).
- A from-scratch Chaitin's-algorithm graph-coloring allocator
  (`codegen/legacy/RegAllocGraphColoring.cpp`, now deleted) that several
  documents in this repository described as the production allocator, but
  that was never actually wired into `codegen/CMakeLists.txt`'s build and
  did not compile as written — `build_liveness_for_block` iterated an
  always-empty vector via an always-false ternary, and `compute_liveness`
  indexed a `std::map<int, LivenessInfo>` with a `std::string` key. It's
  described in detail further down this document for historical interest,
  but it never ran.

Linear scan is a deliberately smaller step up from "no liveness at all"
than graph coloring — well short of Chaitin's algorithm in allocation
quality, but *actually correct and shipping*, which the alternative wasn't.
Upgrading to graph coloring later is a reasonable follow-up; it should
reuse the liveness infrastructure below rather than recompute it, since
that part is architecture-agnostic and already correct.

## How it works today

### 1. Liveness (`codegen/mir/Liveness.cpp`)

Standard two-step construction:

1. **Per-block local use/def sets.** Walk each block's instructions in
   order; a vreg is in `Use[B]` if it's read before any def of it within
   `B`, and in `Def[B]` if it's written anywhere in `B`.
2. **Backward dataflow fixed point** over the block graph:
   `LiveOut[B] = ⋃ LiveIn[S]` for successors `S`;
   `LiveIn[B] = Use[B] ∪ (LiveOut[B] − Def[B])`. Iterated to a fixed point.

This runs over the **real CFG** — `MachineBasicBlock::preds`/`succs`, which
are populated directly from the frontend's `ir::BasicBlock::preds`/`succs`
(computed once by `CFGPass` during frontend compilation) rather than
recomputed. This matters concretely: after `PhiElimination` runs, a
variable that used to be a single SSA value can have *multiple* definition
sites (one `COPY` per incoming edge — see `codegen/mir/PhiElimination.h`'s
doc comment). Naively scanning "first textual def to last textual use"
would either under- or over-shoot badly for a loop-carried variable;
dataflow liveness over the real CFG doesn't have that problem.

Each virtual register's per-block liveness is then flattened into a single
`[start, end]` interval over a block-layout-order instruction numbering —
the standard "fast" linear-scan simplification (no tracking of lifetime
holes within an interval). This never under-approximates a live range (so
it never lets two simultaneously-live values collide in the same physical
register), but can occasionally spill more than a hole-aware allocator
would.

### 2. Allocation (`codegen/mir/LinearScanRegAlloc.cpp`)

Classic Poletto-Sarkar linear scan, run independently per register class
(GPR, FPR):

1. Sort intervals by start point.
2. Walk them in order, maintaining an `active` list of currently-assigned
   intervals sorted by end point.
3. At each new interval: expire `active` entries whose end precedes this
   interval's start (freeing their registers), then look for a free,
   non-busy physical register (see "Physical register constraints" below).
4. If none is free: evict the `active` interval with the **furthest end
   point** if that's later than the current interval's own end (the
   classic `SpillAtInterval` policy) — spilling that one and giving its
   register to the current interval, since the current interval has less
   to lose. Otherwise spill the current interval itself.

Register classes are entirely target-defined
(`TargetRegisterInfo::allocatable(RegClass)`); the allocator itself
contains no architecture-specific logic.

### Physical register constraints

Some physical registers are constrained outside the normal vreg→register
mapping — a `CALL` clobbers every caller-saved register per the ABI, and a
few instruction sequences pin specific registers (e.g. x86's `idiv` reading
`RDX:RAX`). These are tracked as "busy" intervals: any `MachineInstr`
operand with `is_physical == true`, plus each `MachineInstr::clobbers_gpr`/
`clobbers_fpr` entry, marks that physical register unavailable for the
instruction's position. A candidate physical register for a virtual
interval is only offered if it's free for the *entire* interval, not just
at its start — so a value live across a call site correctly can't land in
a caller-saved register.

### Spilling

Two scratch physical registers per class are permanently reserved (the
last two entries of each target's `allocatable()` list) so spill-reload
code can always materialize a spilled operand without contending with the
main allocation — this covers the worst realistic case for this
instruction set (e.g. an x86 read-modify-write instruction with two
distinct spilled operands, such as `dst += rhs` where both `dst`'s reload
and `rhs`'s reload are needed simultaneously). Spilled values get a real
stack frame slot (`MachineFunction::new_frame_object`), resolved to a
concrete offset later by `PrologueEpilogInserter`.

## Verification

Register-pressure correctness is verified by execution, not just
inspection: `codegen/tools/test_scalar_pipeline.cpp` includes a case with
40 simultaneously-live temporaries against roughly 14 allocatable x86 GPRs,
forcing real spill/reload code, and checks the *executed* result is
correct — not just that the assembly looks plausible. See
`REAL_EXECUTION_TESTING_GUIDE.md`.

## Historical note: the dead graph-coloring allocator

An earlier version of this document described
`codegen/legacy/RegAllocGraphColoring.cpp` — liveness dataflow, interference
graph construction, a Chaitin's-algorithm simplify/spill/select coloring
loop — as the production allocator ("the TensorC compiler now uses a
sophisticated graph coloring-based register allocator"). That was never
true: the file was never added to `codegen/CMakeLists.txt`'s build and did
not compile as written (see "Overview" above for the specific bugs). Both
that file and this document's original algorithm write-up have been
removed rather than kept as a stale reference; git history has the original
text if it's useful as a starting point. If graph coloring is implemented
for real in the future, it should live under `codegen/mir/` alongside
`LinearScanRegAlloc.cpp` as an alternative allocator, reusing
`Liveness.cpp`'s dataflow rather than reimplementing it.
