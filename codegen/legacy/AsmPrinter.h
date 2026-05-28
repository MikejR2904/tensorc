#pragma once

/// codegen/AsmPrinter.h
///
/// AsmPrinter — emits GNU assembler (GAS) syntax for RISC-V 64-bit targets.
///
/// After RegAlloc runs, all MachineOperand::is_vreg flags are false and
/// op.reg holds a physical register number.  Operands with reg == SPILLED
/// have been rewritten by RegAlloc before reaching here.
///
/// Output format (GAS / RV64):
///   .text
///   .global <name>
///   <name>:
///   <block_label>:
///     <opcode>\t<op0>, <op1>, ...
///
/// Register naming:
///   GPR:    x0..x31  (or ABI aliases: zero, ra, sp, a0-a7, t0-t6, …)
///   FPR:    f0..f31  (or fa0-fa7, ft0-ft11, …)
///   Vector: v0..v31

#include "MachineFunction.h"
#include "RegAlloc.h"
#include <ostream>
#include <string>

namespace codegen {

struct AsmPrinter {
    std::ostream& os;
    bool use_abi_names = true; // use a0/a1/t0 instead of x10/x11/x5

    explicit AsmPrinter(std::ostream& os, bool abi = true)
        : os(os), use_abi_names(abi) {}

    void print(const MachineFunction& mf);

private:
    std::string gpr_name(int r) const;
    std::string fpr_name(int r) const;
    std::string vec_name(int r) const;
    std::string operand_str(const MachineOperand& op) const;
};

} // namespace codegen
