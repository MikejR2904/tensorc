#pragma once

#include "../TargetInfo.h"

namespace codegen::target::x86_64 {

/// Emits GNU assembler AT&T-syntax text. Deliberately minimal (no
/// .type/.size ELF directives) so the same output assembles unmodified under
/// both ELF (`as` on Linux) and PE-COFF (`as` from a Windows MinGW-w64
/// toolchain, e.g. x86_64-w64-mingw32) — verified against the local
/// toolchain during development (see codegen/tools/x86_64 execution tests).
struct X86AsmPrinter : TargetAsmPrinter {
    void print(std::ostream& os, const MachineFunction& mf, const TargetInstrInfo& tii, const TargetRegisterInfo& tri) override;
};

} // namespace codegen::target::x86_64
