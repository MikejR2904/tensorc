![TensorC Logo](tensorc-vscode/tensorc/icons/tensorc_logo.svg)

# TensorC Compiler Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Stage: Alpha](https://img.shields.io/badge/Stage-Alpha_Development-orange.svg)]()
[![Architecture: RISC-V Target](https://img.shields.io/badge/Target-RISC--V_%2B_Systolic_Array-red.svg)]()

TensorC is an experimental domain-specific compiler for tensor programs. It is designed to optimize tensor workloads for a RISC-V host processor paired with a custom systolic array accelerator. TensorC reads `.tcc` source, validates tensor and type semantics, builds an SSA-based IR, and prepares the compiler pipeline for backend code generation.

---

## Highlights

*   **Hardware-Aware Frontend:** Integrated lexical scanning, recursive-descent parsing, symbolic type analysis, and native SSA basic-block construction.
*   **Regional Loop & Operator Fusion:** A specialized optimization pass that aggregates sequential element-wise operations (e.g., `Add -> Exp -> Div`) into single-pass execution loops (`tensor.fused.elem_chain`) to eliminate intermediate allocations and maximize cache locality.
*   **Decoupled Async Pipelines:** Syntax support for concurrent execution contexts (`async`, `spawn`, `await`), separating host memory setup from background accelerator execution loops.

---

## Build Instructions

### Prerequisites

- `CMake` 3.20 or newer
- C++20-capable compiler
  - MSVC 2022+ on Windows
  - GCC 11+ on Linux
  - Clang 13+ on macOS/Linux
- Internet access for the first build to download GoogleTest

### Build on Windows / MSVC

```powershell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j
```

### Build on Unix-like systems

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j
```

### Output

- `build/bin/tensorc.exe` on Windows
- `build/bin/tensorc` on Linux/macOS

> If the build fails because warnings are treated as errors, fix the reported warnings or temporarily adjust compiler warning settings in `CMakeLists.txt`.

---

## CLI Usage

The compiler driver is implemented in `cli/tensorc.cpp`.

```text
Usage: tensorc <file.tcc> [options]
Options:
  -h, --help      Show this help message
  -v, --version   Display compiler version
  --print-ir      Print the generated IR after compilation
```

Example:

```powershell
build/bin/tensorc cli/test_files/fused_example.tcc --print-ir
```

---

## Language Reference

TensorC source files use the `.tcc` extension and support a compact tensor DSL.

### Modules and imports

Currently, we are supporting built-in modules such as tensor, nn, optim, data, and parallel. Example usage as follows. 

```text
import tensor as ts;
```

Documentation on the built-in modules and how to construct custom modules to be included in the future iterations.

### Function declarations

```text
async fn fused_example(x: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32> {
    let intermediate = x @ w;
    return ts::relu(intermediate);
}
```

### Bindings

- `let x = expr;`
- `let y: Tensor<f32> = expr;`
- `let grad z = expr;` for gradient-aware declarations
- `fn name#(T, U)(...) -> ...` generic functions

### Control flow

- `if`, `else`
- `for` loops
- `while` loops
- `match` statements
- `break`, `continue`

### Expressions and returns

- `@` is used for tensor contraction and fusion-style operators.
- `ts::relu(...)` shows namespaced builtin calls.
- A compound block may return its final expression implicitly when no explicit `return` is present.

### Example

```text
import tensor as ts;

async fn fused_example(x: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32> {
    let intermediate = x @ w;
    return ts::relu(intermediate);
}
```

---

## Testing

Build and run the integrated test suite:

```powershell
cmake --build build --config Release --target tensorc_tests
cd build
ctest --output-on-failure -C Release
```

Test artifacts are available under `build/bin`.

---

## Repository Layout

- `cli/` — compiler frontend entrypoint
- `compiler/` — lexer, parser, AST, semantic analyzer, IR builder
- `compiler/lexer/` — tokenization implementation
- `compiler/parser/` — recursive descent parser and grammar
- `compiler/ast/` — symbol table, type tracking, and semantic validation
- `compiler/ir/` — IR nodes, passes, and printing
- `runtime/` — tensor runtime foundations
- `tests/` — unit tests for compiler internals

---

## Contributing

Contributions are welcome. Please:

- Open issues for bugs and feature requests
- Submit clear pull requests
- Keep changes focused and testable
- Respect the existing C++20 code style

---

## License

TensorC is released under the [MIT License](LICENSE).
