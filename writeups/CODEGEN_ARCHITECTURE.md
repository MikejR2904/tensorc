# TensorC Code Generation Architecture

## System Design

TensorC's backend is two independent pipelines sharing one entry point
(`codegen::CodegenDriver`):

```
                ┌─────────────────────────────────────────────┐
                │         IR Module (compiler/ir/*)            │
                │    (Scalar Instructions + Tensor Ops)        │
                └────────────────┬──────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                  │
        [Scalar / control-flow]           [Tensor operations]
     ScalarCodegenPipeline               TensorOpLowering bridge +
     (codegen/mir/, codegen/target/)     progressive lowering
                │                          (codegen/bridge/,
                │                           codegen/lowering/,
                │                           codegen/targets/)
                ▼                                  ▼
         x86-64 / RISC-V64                 x86-64 / RISC-V64
         real machine code                 (matmul family only;
                                            RISC-V targets a
                                            speculative custom
                                            accelerator ISA, not
                                            real hardware)
```

This document covers the **scalar pipeline** — the part that changed. For
the tensor pipeline (untouched by that work, and with real limitations of
its own), see `codegen/README.md`'s "Tensor pipeline" section and
`writeups/PROGRESSIVE_LOWERING.md`.

## Why a rewrite happened

The previous scalar backend (`codegen/legacy/`, now deleted) had:

- **String-mnemonic Machine IR.** `MachineInstr::opcode` was a raw RV64
  mnemonic string (`"add"`, `"fadd.d"`) chosen the moment instruction
  selection ran — there was no target-independent layer, so supporting a
  second architecture meant forking the whole pipeline, not adding a
  selection stage.
- **No liveness analysis.** The register allocator treated every virtual
  register as live for the entire function and spilled far more than
  necessary; the register allocator that was *documented* as production —
  a from-scratch Chaitin's-algorithm graph-coloring implementation — was
  never wired into the CMake build and didn't compile as written (a
  `std::map<int,...>` indexed by `std::string`, among other issues).
- **Unsound phi lowering.** PHI nodes were lowered to `mv` instructions
  emitted in the phi's own block, which its own comment called "a
  conservative approximation" — unsound in general (the classic lost-copy
  problem on critical edges and parallel copies).
- **Informal ABI.** Calling convention was hardcoded to "first 8 integer
  args in a0-a7", no stack-passed-argument support, no callee-saved
  register handling, no real frame pointer chain.

## Current architecture: the scalar pipeline

```
ir::Function
    │  GenericISel               (compiler SSA-IR -> target-independent Machine IR)
    ▼
MachineFunction (generic)         codegen/mir/MachineIR.h — MOp enum, virtual
    │                             registers, real CFG edges reused directly
    │  PhiElimination             from the frontend's CFGPass (no recomputation)
    ▼
MachineFunction (SSA-destroyed)
    │  CallLowering               ABI-independent code, parameterized by
    ▼                             CallingConv — assigns real physical-register
MachineFunction (ABI copies       or stack locations to args/returns
    inserted as plain COPY/
    LOAD/STORE)
    │  TargetInstrInfo::select()  generic MOp -> real target opcodes
    ▼                             (x86-64 or RISC-V64)
MachineFunction (target instrs)
    │  LinearScanRegAlloc         real liveness (backward CFG dataflow) ->
    ▼                             Poletto-Sarkar linear scan
MachineFunction (physical regs
    + spill code)
    │  PrologueEpilogInserter     stack-frame layout, callee-saved
    ▼                             save/restore, frame-pointer chain
MachineFunction (final)
    │  TargetAsmPrinter
    ▼
Assembly text
```

Every stage from `TargetInstrInfo::select()` onward is entirely
target-independent — it only calls through four interfaces defined in
`codegen/target/TargetInfo.h`:

| Interface | Responsibility |
|---|---|
| `TargetInstrInfo` | generic opcode → target opcode selection; `MCInstrDesc` table (mnemonic + flags) per target opcode |
| `TargetRegisterInfo` | physical register enumeration, register classes, callee-saved/caller-saved sets, ABI stack-argument offset |
| `CallingConv` | classifies each argument/return value to a register or stack location |
| `FrameLowering` | the actual prologue/epilogue *instruction shape* — genuinely different per architecture (x86-64 `push`/`pop` + `sub`/`add rsp` vs. RISC-V's `addi sp,sp,-N` + explicit `ra` save, since RISC-V has no auto-pushed return address), so this can't be table-driven the way instruction selection is (mirrors LLVM's `TargetFrameLowering` split for the same reason) |

Two targets implement these today: `codegen/target/x86_64/` (System V AMD64
ABI, AT&T syntax) and `codegen/target/riscv64/` (RV64GC, LP64D ABI). Adding
a third (e.g. ARM64) means implementing these four interfaces under
`codegen/target/<name>/` — nothing in `codegen/mir/` needs to change.

### The Machine IR itself

`codegen/mir/MachineIR.h` defines one `MachineInstr` type used for both
generic and target-specific instructions (`is_generic` flag distinguishes
them) — closer to how LLVM's own `MachineInstr` class works (one class,
opcode meaning changes) than to GlobalISel's separate generic/target split.
Operands (`MOperand`) can be a virtual or physical register, an immediate,
a frame-slot reference, a global symbol, or a basic-block reference. No
immediate folding is implemented — every constant is materialized into its
own vreg by `GenericISel` (the `-O0`-equivalent baseline); folding an
immediate directly into e.g. `addq $imm, dst` is a straightforward
follow-up peephole, not required for correctness.

### Register allocation

`codegen/mir/Liveness.h/.cpp` computes real liveness: per-block
use/def sets, then a standard backward dataflow fixed point over the actual
CFG edges (reused from the frontend's `CFGPass`, not recomputed), then
flattened into one `[start, end]` interval per virtual register.
`codegen/mir/LinearScanRegAlloc.h/.cpp` runs classic Poletto-Sarkar linear
scan over those intervals — a deliberately smaller step up from "no
liveness at all" than graph coloring, but one that's actually correct and
shipping rather than more sophisticated on paper. It also tracks which
physical registers are pinned by ABI copies or a `CALL`'s clobber set (so a
value live across a call site can't be assigned a caller-saved register)
and which callee-saved registers actually got used, for
`PrologueEpilogInserter` to save/restore.

### Verification

This isn't just "it compiles" — see `REAL_EXECUTION_TESTING_GUIDE.md` at
the repo root. Generated x86-64 code is assembled, linked, and *executed*
via the local `x86_64-w64-mingw32` toolchain, with results checked against
independently-computed expected values, for both hand-built IR (precise
control over specific MIR patterns like critical-edge phi elimination and
register-pressure spilling) and real `.tcc` source compiled through the
full frontend. That real-source testing caught five bugs — a segfault, a
silently-miscompiled loop pattern, and three distinct backend bugs — that
hand-built-IR testing alone had missed; see that document's "What this has
already caught" section.

## Known gaps (scalar pipeline)

- No string/global-constant materialization (no `.rodata` emission yet) —
  a call argument that's a string literal gets a vreg with no defining
  instruction.
- ARM64 and other targets aren't implemented (though the interface split
  above is designed for it).
- RISC-V64 execution isn't verified end-to-end (no cross-assembler/emulator
  available in this environment) — only structural checks.
