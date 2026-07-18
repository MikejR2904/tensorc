# TensorC Code Generation Module

Two independent pipelines: a target-independent scalar/control-flow backend (real Machine IR, liveness-based register allocation, and a real ABI, for x86-64 and RISC-V64), and a separate progressive tensor-kernel lowering pipeline.

## Quick Start

```cpp
#include "codegen/CodegenDriver.h"

CodegenDriver driver("x86_64"); // or "riscv64"

// Scalar/control-flow function
driver.lower_scalar_function(fn, "output.s");

// Tensor operations (unrelated pipeline, see below)
std::map<const void*, TensorShape> shapes = {{a, {128, 256, 8}}};
driver.lower_tensor_operation(matmul_op, shapes, std::cout);
```

## Architecture

**Scalar pipeline** (`codegen/mir/`, `codegen/target/`) — GenericISel lowers frontend SSA-IR into a target-independent Machine IR (`codegen::mir::MachineFunction`, opcodes in `MOp`, modeled on LLVM's MachineInstr rather than target-specific mnemonic strings), then:

```
GenericISel → PhiElimination → CallLowering → TargetInstrInfo::select()
  → LinearScanRegAlloc → PrologueEpilogInserter → TargetAsmPrinter
```

Every stage after `select()` is entirely target-independent — it only calls through the `TargetInstrInfo` / `TargetRegisterInfo` / `CallingConv` / `FrameLowering` interfaces in `codegen/target/TargetInfo.h`. Adding a target (e.g. ARM64) means implementing those four interfaces under `codegen/target/<name>/`, not touching `codegen/mir/` at all. Two targets exist today: `codegen/target/x86_64/` (System V AMD64 ABI, AT&T syntax) and `codegen/target/riscv64/` (RV64GC, LP64D ABI).

**Tensor pipeline** (`codegen/bridge/`, `codegen/lowering/`, `codegen/targets/`) — a separate, untouched 4-phase progressive lowering pipeline (`Tiler → ScratchpadAllocator → MemoryLegalizer → Scheduler → {X86,RiscV}TargetEmitter`) for `TensorOpInst`. Its RISC-V output targets a speculative custom systolic-array accelerator ISA (`.insn`-encoded `AME_COMPUTE`/`DMA_LOAD`/`DMA_STORE` on RISC-V's custom-0 opcode) modeled after research literature, not real hardware — see `codegen/targets/README.md`. `CodegenDriver::lower_module_to_asm` splices this pipeline's output in alongside the scalar pipeline's.

## Directory Structure

```
codegen/
├── CodegenDriver.h/cpp          # Unified entry point: scalar functions + tensor-op splicing
├── ScalarCodegenPipeline.h/cpp  # Orchestrates the scalar pipeline stages listed above
├── mir/                         # Target-independent Machine IR + passes
│   ├── MachineIR.h               # MOp, LLT, MOperand, MachineInstr/Function/BasicBlock
│   ├── GenericISel.h/cpp         # ir::Instruction -> generic MachineInstr
│   ├── PhiElimination.h/cpp      # SSA destruction, incl. critical-edge splitting
│   ├── CallLowering.h/cpp        # ABI-independent code, parameterized by CallingConv
│   ├── Liveness.h/cpp            # Real CFG-based liveness (backward dataflow)
│   ├── LinearScanRegAlloc.h/cpp  # Poletto-Sarkar linear scan over that liveness
│   └── PrologueEpilogInserter.h/cpp
├── target/
│   ├── TargetInfo.h/cpp          # The interfaces + create_target() factory
│   ├── x86_64/                   # System V AMD64
│   └── riscv64/                  # RV64GC / LP64D
├── bridge/, lowering/, targets/  # Tensor-kernel pipeline (see above; unrelated to mir/target)
└── tools/                        # Tests, incl. tools/ExecUtils.* (x86-64 assemble+link+execute harness)
```

## Testing

```bash
cmake --build build --target test-real-execution codegen-scalar-pipeline-test codegen-progressive-test
./build/bin/test-real-execution              # Real .tcc *source* -> full frontend -> ScalarCodegenPipeline
                                              # -> assembled, linked, and *executed* via the local toolchain,
                                              # checked against independently-computed expected values.
                                              # Exercises the parser/IRBuilder too, not just codegen — see
                                              # REAL_EXECUTION_TESTING_GUIDE.md for what this has already caught.
./build/bin/codegen-scalar-pipeline-test     # Same execution-verification approach, but against hand-built
                                              # ir::Function objects for precise control over specific MIR
                                              # patterns (critical-edge phi elimination, register-pressure
                                              # spilling). RISC-V64 output only gets structural assertions in
                                              # both suites — no cross-assembler or emulator is available here.
./build/bin/codegen-progressive-test         # Tensor pipeline structural tests
```

See `REAL_EXECUTION_TESTING_GUIDE.md` at the repo root for the full picture,
including which bugs this approach has already caught.

## Known gaps (scalar pipeline)

- No support yet for string/global-constant materialization (no `.rodata`/`.data` section emission) — a call argument that's a string literal gets a vreg with no defining instruction. Same gap the pipeline it replaced had; not yet in scope.
- `TensorOpInst` is untouched — still routed through the separate tensor pipeline above.
- ARM64 and other targets aren't implemented, though the `TargetInfo.h` interfaces are designed so adding one doesn't require touching `codegen/mir/`.
- No `%` (modulo) operator in the language grammar — `ir::BinOpCode::Mod` lowers correctly (verified via hand-built IR) but no `.tcc` source can reach it.
