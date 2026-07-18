#pragma once

/// codegen/tools/ExecUtils.h
///
/// Minimal x86-64 execution-verification harness: assembles generated AT&T
/// text with the real local toolchain (confirmed present:
/// x86_64-w64-mingw32 gcc/as/ld), links it into a shared library, and loads
/// it so a test can call the compiled function directly and check a real
/// numeric result — replacing string-content-only assertions like
/// codegen/tools/test_legacy_scalar.cpp's `has_mnemonic("add")`.
///
/// Deliberately NOT the full REAL_EXECUTION_TESTING_GUIDE.md harness (no ELF
/// parsing, no cross-target support) — scoped to proving the new x86-64
/// scalar backend correct, since that's the only target this machine can
/// actually assemble+link+execute (no riscv64 cross-toolchain or emulator is
/// installed here; see the backend redesign plan's verification section).

#include <string>

namespace codegen::testing {

/// Assembles `asm_text` and links it into "<work_dir_prefix>.dll". Returns
/// an opaque module handle (nullptr on failure — check "<work_dir_prefix>.log"
/// for the assembler/linker's stderr).
void* assemble_and_load(const std::string& asm_text, const std::string& work_dir_prefix);

void* get_symbol(void* module, const std::string& name);

void unload(void* module);

} // namespace codegen::testing
