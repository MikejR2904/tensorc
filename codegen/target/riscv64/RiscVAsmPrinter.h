#pragma once

#include "../TargetInfo.h"

namespace codegen::target::riscv64 {

/// Emits standard RV64GC GNU-assembler text (ABI register names, `.insn`-free
/// — real mnemonics only, unlike codegen/targets/RiscVTargetEmitter.cpp's
/// custom-accelerator `.insn` encoding, which is a separate, untouched
/// pipeline for tensor kernels).
struct RiscVAsmPrinter : TargetAsmPrinter {
    void print(std::ostream& os, const MachineFunction& mf, const TargetInstrInfo& tii, const TargetRegisterInfo& tri) override;
};

} // namespace codegen::target::riscv64
