# Real Execution Testing

**Status**: Implemented and running.

## Problem statement (why this exists)

Several of the compiler's older test suites (`codegen/tools/test_builtin_modules.cpp`,
`codegen/tools/test_execution_examples.cpp`) verify builtin math/tensor
behavior by comparing a C++ reference implementation against itself:

```cpp
// This doesn't test the compiler at all — it tests that std::sin equals std::sin.
TEST_F(MathModuleTest, Sin) {
    double expected = std::sin(x);
    double actual = std::sin(x);
    EXPECT_NEAR(actual, expected, eps);   // Can never fail.
}
```

Those tests aren't wrong to keep (they document expected numeric behavior),
but they can't catch a single compiler bug, because no TensorC source is ever
parsed, no IR is ever built, and no generated code is ever run. This
document describes the pipeline that closes that gap: real `.tcc` source
text, compiled through the actual compiler, assembled and linked with a real
toolchain, executed, and checked against an independently-computed expected
value.

**This isn't a hypothetical concern.** The first time this pipeline was
pointed at real programs instead of hand-built IR, it caught five
previously-undetected bugs in a single session — a segfault, a silent
miscompilation of loop accumulators, two backend register-width bugs, and a
linker-symbol mismatch. See "What this has already caught" below.

## Architecture: what's actually implemented

```
TensorC source text (std::string, not necessarily a file on disk)
    │  Lexer → Parser → SemanticAnalyzer → IRBuilder::build() → PassPipeline::run()
    ▼
ir::IRModule                                    (compiler/ir/*)
    │  ScalarCodegenPipeline::lower_function() per function
    ▼
x86-64 assembly text (AT&T syntax)               (codegen/*)
    │  gcc -shared  (invokes the local `as` + `ld`)
    ▼
A loaded .dll                                    (Windows PE, this environment)
    │  GetProcAddress + a typed, sysv_abi function-pointer cast
    ▼
Call the function on real hardware, compare the return value to an
independently computed expected value
```

Two files make this a three-line operation for a test:

- **`codegen/tools/CompilationTestUtils.h`** — `compile_source_to_asm(source, target)`
  runs the *exact* sequence `cli/tensorc.cpp` uses (Lexer → Parser →
  SemanticAnalyzer → IRBuilder → PassPipeline → ScalarCodegenPipeline) against
  a source string and returns the resulting assembly text. This replaces an
  earlier draft of this same idea that invented API names
  (`Lexer::tokenize`, `IRBuilder::build(ast)` with no module,
  `CodegenDriver::generate_assembly`) that never matched the real compiler
  and was never implemented.
- **`codegen/tools/ExecUtils.h`** — `assemble_and_load(asm_text, prefix)`
  writes the text to `<prefix>.s`, runs `gcc -shared -o <prefix>.dll
  <prefix>.s` (the local `x86_64-w64-mingw32` toolchain — confirmed present
  and working in this environment), and `LoadLibrary`s the result.
  `get_symbol`/`unload` wrap `GetProcAddress`/`FreeLibrary`.

A test using both looks like:

```cpp
std::string asm_text = codegen::testing::compile_source_to_asm(
    "fn add_i32(a: i32, b: i32) -> i32 { return a + b; }", "x86_64");

void* mod = codegen::testing::assemble_and_load(asm_text, "tests/results/scratch");
typedef int32_t (*sysv_i32_i32i32)(int32_t, int32_t) __attribute__((sysv_abi));
auto* f = reinterpret_cast<sysv_i32_i32i32>(codegen::testing::get_symbol(mod, "add_i32"));

EXPECT_EQ(f(17, 25), 42);   // Runs on real hardware, not a reference implementation.
codegen::testing::unload(mod);
```

The `__attribute__((sysv_abi))` on the function-pointer typedef matters: this
compiler targets the **System V AMD64 ABI** (the standard, portable x86-64
calling convention, not tied to any OS) because that's what real x86-64
tooling generally assumes. This development environment's default C ABI is
the *Windows x64* calling convention, which differs (different argument
registers, different shadow-space requirements). `sysv_abi` is a real
GCC/Clang function-attribute extension that tells the caller "marshal
arguments for System V, not Windows x64" — it's not a hack, and it means the
backend doesn't need a Windows-specific ABI just to be testable here.

### Why "PE .dll" and not "ELF" (adjusting an earlier assumption)

An earlier draft of this pipeline assumed ELF binaries and `dlopen`/`dlsym`
(the Linux/macOS convention). This environment is Windows, so the actual
implementation uses `LoadLibraryA`/`GetProcAddress` against a PE `.dll`
produced by `gcc -shared` — functionally equivalent for this purpose (load a
compiled function, get a pointer, call it), just the Windows API instead of
the POSIX one. `ExecUtils.cpp` is `#ifdef`-free and Windows-only right now;
a POSIX backend (`dlopen`/`dlsym`, `.so`/`.dylib`) would be a small, separate
addition for Linux/macOS CI, not a redesign — the `assemble_and_load` /
`get_symbol` / `unload` interface in `ExecUtils.h` doesn't need to change.

## What's tested today

- **`codegen/tools/test_real_execution.cpp`** (11 tests) — real `.tcc`
  *source text* covering i32/i64/f64 arithmetic, comparisons, `&&`/`||`
  boolean logic, `while` loops (including multi-accumulator patterns like
  Fibonacci), recursion, and multi-function call chains. This is the suite
  that exercises the full pipeline including the parser and `IRBuilder` —
  see "What this has already caught" below for why that distinction mattered
  in practice.
- **`codegen/tools/test_scalar_pipeline.cpp`** (42 checks) — hand-built
  `ir::Function` objects (bypassing the parser) exercising specific
  MIR-level patterns precisely: critical-edge phi elimination, register
  spilling under deliberate pressure (40 simultaneously-live temporaries
  against ~14 allocatable x86 GPRs), and cross-function ABI calls. Precise
  control over IR shape at the cost of not exercising the frontend.

Both build via CMake (`codegen-scalar-pipeline-test`, `test-real-execution`
targets; the latter is also registered with `gtest_discover_tests`) and are
Windows-only for the reason above.

```bash
cmake --build build --target test-real-execution codegen-scalar-pipeline-test
./build/bin/test-real-execution.exe
./build/bin/codegen-scalar-pipeline-test.exe
```

## What's out of scope (and why, honestly)

- **Builtin modules (`math::`, `tensor::`, `nn::`, ...) don't link.**
  `math::sin(x)` lowers to a `CallInst` targeting the mangled symbol
  `math.sin` (see `compiler/ir/ir_modules/math_handler.cpp`) — not the real
  libm symbol `sin` — so it compiles to a `call math.sin` instruction with no
  matching definition anywhere. This isn't a gap in the execution-testing
  pipeline; it's a real gap in the compiler (no runtime/shim library exists
  yet for builtin-module calls). `test_builtin_modules.cpp` /
  `test_execution_examples.cpp` remain reference-only for this reason.
- **Tensor ops (`TensorOpInst`) are untouched** — they go through a separate
  progressive-lowering pipeline (`codegen/bridge/`, `codegen/lowering/`,
  `codegen/targets/`) targeting a speculative custom accelerator ISA on
  RISC-V, not real hardware. See `codegen/README.md`.
- **RISC-V64 execution isn't verified**, only structural (mnemonic presence)
  checks — no RISC-V cross-assembler or emulator is available in this
  environment. If that's needed later, it requires installing one (a
  `riscv64-*-gcc` cross toolchain or `qemu-riscv64`), which wasn't done here
  since it modifies the system outside the scope of this work.
- **`%` (modulo) has no surface syntax** — `ir::BinOpCode::Mod` exists and the
  backend lowers it correctly (verified via hand-built IR in
  `test_scalar_pipeline.cpp`), but the lexer/parser have no token for it, so
  no `.tcc` source can reach it. Not a testing-pipeline gap — a language
  grammar gap.

## What this has already caught

The bugs below were found the first time this pipeline ran against real
multi-statement, multi-function source — none were visible to hand-built-IR
testing, because hand-built IR sidesteps the exact code paths that were
broken.

1. **Segfault on unresolved module calls** — `IRBuilder::lower_call`'s
   stub-function fallback could construct a `CallInst` with a null `callee`
   whenever no `IRModule` was attached, and the constructor's `track_uses()`
   dereferenced it unconditionally. Fixed with an explicit check that throws
   a clear error instead. (`compiler/ir/IRBuilder.h`)
2. **Loop accumulators were silently wrong** — `let`-bound scalars were
   promoted to a stack slot lazily, on their *first write*. A loop body like
   `s = s + i;` reads `s` before that write's promotion runs, so the read
   captured the pre-loop value as a fixed IR operand forever — every
   iteration recomputed the same constant instead of accumulating. Every
   `while`/`for` loop with an accumulator or counter was affected. Fixed by
   promoting scalar `let` bindings to a stack slot eagerly.
   (`compiler/ir/IRBuilder.h`, `lower_let`)
3. **i32 arithmetic produced invalid assembly** — the x86-64 backend's
   register-to-register move/RMW helper hardcoded 64-bit register names
   regardless of the operand's actual width, generating instructions like
   `mov %eax, %rdx` (mismatched operand sizes) that GNU `as` rejects. Never
   caught before because the hand-built-IR test suite happened to only use
   i64 values. (`codegen/target/x86_64/X86InstrInfo.cpp`)
4. **Function calls between real TensorC functions couldn't link** — function
   *definitions* kept the IR's internal `@` symbol-name sigil
   (`.globl @foo`), but *call sites* stripped it (`call foo`) — a mismatch
   invisible to hand-built IR, which named functions directly without the
   sigil. (`codegen/mir/GenericISel.cpp`)
5. **Boolean `&&`/`||`/`!` could read garbage** — `AND`/`OR`/`XOR`/`NOT` of
   two 1-bit boolean values were emitted as 8-bit register operations; x86
   doesn't auto zero-extend an 8-bit register write the way it does a 32-bit
   one, so stale data from whatever previously occupied the upper bits of
   that physical register survived and could make a `testq`-based branch
   take the wrong path. Fixed by operating on booleans at 32-bit width
   instead (safe, since every boolean-producing instruction already
   guarantees zero-extension at that width).
   (`codegen/target/x86_64/X86InstrInfo.cpp`)

## Extending this

To add a new real-execution test case: write the `.tcc` source as a
`std::string` literal, pick (or add) a `typedef ... __attribute__((sysv_abi))`
matching its signature, and follow the pattern in
`codegen/tools/test_real_execution.cpp`. Keep new cases scoped to what
`ScalarCodegenPipeline` actually supports (scalar arithmetic, control flow,
function calls) until the tensor pipeline and a builtin-module runtime exist.
