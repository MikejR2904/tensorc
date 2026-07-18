#pragma once

/// codegen/target/TargetInfo.h
///
/// The retargetability seam. Every backend (x86-64, RISC-V64, and any target
/// added later) implements TargetInstrInfo + TargetRegisterInfo + CallingConv
/// + TargetAsmPrinter and is selected purely by name through create_target().
/// Nothing above this header (Liveness, LinearScanRegAlloc, CallLowering,
/// PrologueEpilogInserter) contains any per-architecture logic — they only
/// call through these interfaces. This replaces the ad hoc
/// `target_ == "x86_64"` string checks and RV64-only mnemonic strings in the
/// old codegen/legacy pipeline.

#include "../mir/MachineIR.h"
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace codegen::target {

using mir::LLT;
using mir::MachineFunction;
using mir::MachineBasicBlock;
using mir::MachineInstr;
using mir::MOperand;
using mir::MOp;
using mir::RegClass;
using mir::CmpPred;

/// Static description of one target-specific opcode. Replaces the old
/// backend's string-matched opcode checks (e.g. legacy AsmPrinter/RegAlloc's
/// `first_operand_is_def()` hardcoding "sd"|"fsd"|"j"|...).
struct MCInstrDesc {
    const char* mnemonic = "";
    bool operand0_is_def = true;
    bool may_load = false;
    bool may_store = false;
    bool is_terminator = false;
    bool is_call = false;
    bool is_return = false;
    /// Pseudo markers (e.g. frame-setup placeholders) carry no direct asm
    /// text of their own; AsmPrinter / PrologueEpilogInserter handle them.
    bool is_pseudo = false;
};

/// Lowers one generic MachineInstr into 1..N target MachineInstrs
/// (is_generic == false, target_op indexing describe()). May allocate fresh
/// vregs via mf.new_vreg() for legalization (e.g. materializing a 64-bit
/// immediate, or a compare+setcc sequence on x86). Never introduces new
/// basic blocks — selection is entirely intra-block for TensorC's current
/// instruction set (see MachineIR.h).
struct TargetInstrInfo {
    virtual ~TargetInstrInfo() = default;
    virtual std::vector<MachineInstr> select(const MachineInstr& generic, MachineFunction& mf) = 0;
    virtual const MCInstrDesc& describe(unsigned target_op) const = 0;

    /// Opcode LinearScanRegAlloc should use to spill/reload a register of the
    /// given class to/from a frame slot (always full stack-slot width — no
    /// narrow spills in this phase). The only target-specific knowledge
    /// RegAlloc needs; everything else it does is architecture-agnostic.
    virtual unsigned spill_load_opcode(RegClass) const = 0;
    virtual unsigned spill_store_opcode(RegClass) const = 0;
};

struct TargetRegisterInfo {
    virtual ~TargetRegisterInfo() = default;
    /// Allocatable physical registers for a class, in allocation-preference order.
    virtual const std::vector<int>& allocatable(RegClass) const = 0;
    virtual const std::vector<int>& callee_saved(RegClass) const = 0;
    virtual const std::vector<int>& caller_saved(RegClass) const = 0;
    virtual int sp() const = 0;
    virtual int fp() const = 0;          // frame-pointer physreg
    virtual int ra() const = 0;          // link-register physreg, or -1 if the target has none (x86: return addr lives on the stack)
    /// `width` picks the sub-register alias to print (e.g. x86 rax/eax/ax/al
    /// for the same physical GPR); defaults to the class's natural full width.
    virtual std::string reg_name(RegClass, int physreg, LLT width = LLT::I64) const = 0;
    virtual int stack_alignment() const = 0; // bytes
    virtual int word_size() const = 0;       // bytes (8 for both current targets)
    /// Offset of the first incoming stack-passed argument, relative to FP.
    /// x86-64: 16 (8-byte return address pushed by `call` + 8-byte pushed
    /// old FP both sit between FP and the caller's stack args). RISC-V64: 0
    /// (the link register isn't auto-pushed — FP is set to the pre-call SP,
    /// which is exactly where the caller placed the first stack argument).
    virtual int64_t incoming_args_base_offset() const = 0;
};

/// Where one argument or the return value lives, per the target's ABI.
struct ArgLocation {
    bool in_register = true;
    int physreg = -1;
    RegClass rclass = RegClass::GPR;
    int64_t stack_offset = 0;  // valid when !in_register; offset from SP at the call site
};

struct CallingConv {
    virtual ~CallingConv() = default;
    virtual std::vector<ArgLocation> classify_args(const std::vector<LLT>& arg_types) const = 0;
    virtual ArgLocation classify_return(LLT ret_type) const = 0;
};

/// Everything PrologueEpilogInserter (target-independent: pure offset
/// arithmetic) needs to hand off to a target's actual prologue/epilogue
/// instruction *shape*, which genuinely differs per architecture (x86-64
/// push/pop + sub/add rsp vs. RISC-V's addi-sp + explicit ra save) — not
/// just different mnemonics for the same shape, so this can't be
/// table-driven the way instruction selection is. Mirrors LLVM's
/// TargetFrameLowering split for the same reason.
struct SavedReg { int physreg; RegClass rclass; int64_t offset; }; // offset relative to FP

struct FrameLayout {
    int64_t frame_size = 0;   // total bytes reserved below FP (already stack-aligned)
    bool save_ra = false;     // RISC-V: explicit link-register save; x86: ignored (ra is implicit via call/ret)
    int64_t ra_offset = 0;    // valid iff save_ra
    std::vector<SavedReg> saved_callee_regs; // excludes the FP-chain register itself
};

struct FrameLowering {
    virtual ~FrameLowering() = default;
    virtual std::vector<MachineInstr> emit_prologue(const FrameLayout&) const = 0;
    virtual std::vector<MachineInstr> emit_epilogue(const FrameLayout&) const = 0;
};

struct TargetAsmPrinter {
    virtual ~TargetAsmPrinter() = default;
    virtual void print(std::ostream& os, const MachineFunction& mf,
                        const TargetInstrInfo& tii, const TargetRegisterInfo& tri) = 0;
};

/// Bundles one target's four components. ScalarCodegenPipeline holds one of
/// these and never branches on architecture itself.
struct Target {
    virtual ~Target() = default;
    virtual const char* name() const = 0;
    virtual TargetInstrInfo& instr_info() = 0;
    virtual TargetRegisterInfo& reg_info() = 0;
    virtual CallingConv& calling_conv() = 0;
    virtual TargetAsmPrinter& asm_printer() = 0;
    virtual FrameLowering& frame_lowering() = 0;
};

/// Factory: "x86_64" | "riscv64". Returns nullptr for an unknown name.
std::unique_ptr<Target> create_target(const std::string& name);

} // namespace codegen::target
