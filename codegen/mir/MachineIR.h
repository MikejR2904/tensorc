#pragma once

/// codegen/mir/MachineIR.h
///
/// Target-independent Machine IR ("MIR"). Modeled on LLVM's generic
/// MachineInstr (GlobalISel-style generic opcodes) but flattened: TensorC's
/// current scalar instruction set is small enough that a separate legalizer
/// pass isn't needed yet — legalization (e.g. splitting a 64-bit immediate,
/// synthesizing a compare+branch sequence) happens inside each target's
/// instruction selector (see codegen/target/TargetInfo.h).
///
/// This header must not depend on compiler/ir/*  — MIR is produced from the
/// frontend SSA-IR by GenericISel.cpp, but is otherwise a self-contained
/// representation any target backend can consume.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace codegen::mir {

/// Target-independent opcodes. Operands are always virtual registers,
/// immediates, frame indices, globals, or block references — never physical
/// registers (those only appear after CallLowering inserts ABI copies, and
/// even then via MOperand::PhysReg, still within this same generic set).
enum class MOp {
    ADD, SUB, MUL, SDIV, UDIV, SREM, UREM,
    AND, OR, XOR, SHL, LSHR, ASHR,
    NEG, NOT,
    FADD, FSUB, FMUL, FDIV, FNEG,
    ICMP, FCMP,              // predicate carried via a Pred-kind use operand
    LOAD, STORE,
    MOV_IMM, FMOV_IMM, COPY,
    BR, CBR, RET, CALL, PHI,
    FRAME_ADDR,               // materialize the address of a frame object into a vreg
};

/// "Low level type" — just enough to pick a register class and instruction
/// width; not a full type system (that's compiler/ast/Type.h's job upstream).
enum class LLT : uint8_t { I1, I8, I16, I32, I64, F32, F64, Ptr };

inline bool is_float_llt(LLT t) { return t == LLT::F32 || t == LLT::F64; }

enum class RegClass : uint8_t { GPR, FPR };
inline RegClass reg_class_for(LLT t) { return is_float_llt(t) ? RegClass::FPR : RegClass::GPR; }

enum class CmpPred { IEQ, INE, SLT, SLE, SGT, SGE, ULT, ULE, UGT, UGE, FEQ, FNE, FLT, FLE, FGT, FGE };
inline bool is_float_pred(CmpPred p) { return p >= CmpPred::FEQ; }

struct MachineBasicBlock;

/// A single MIR operand. Registers are virtual until CallLowering /
/// instruction selection pins down physical ones (is_physical == true);
/// RegAlloc treats is_physical operands as pre-colored and never rewrites
/// them, but does account for them when computing free-register sets.
struct MOperand {
    /// Mem covers both register-relative addressing (base = `vreg`/
    /// `is_physical`, displacement = `imm`, `frame_idx == -1`) and
    /// frame-relative addressing (`frame_idx >= 0`; the base register is
    /// implicit — SP or FP, chosen per-target by PrologueEpilogInserter,
    /// which is also what resolves `frame_idx` into a concrete `imm`
    /// displacement once the final frame layout is known). Used directly by
    /// spill code and outgoing stack-argument code; general loads/stores
    /// through a user pointer always go through the register-relative form.
    enum Kind { Reg, Imm, FImm, FrameIdx, Mem, Global, Block } kind = Reg;

    // Kind::Reg, and the base register of Kind::Mem when frame_idx == -1
    int vreg = -1;
    bool is_physical = false;   // true => `vreg` field holds a target physreg number
    RegClass rclass = RegClass::GPR;
    LLT type = LLT::I64;        // value type (Reg/Imm/FImm) or access width (Mem)

    // Kind::Imm / FImm, and the displacement of Kind::Mem
    int64_t imm = 0;
    double fimm = 0.0;

    // Kind::FrameIdx (address-of), and the frame-relative form of Kind::Mem
    int frame_idx = -1;

    // Kind::Global (direct-call / symbol reference)
    std::string global;

    // Kind::Block (branch target)
    MachineBasicBlock* block = nullptr;

    static MOperand Reg_(int v, LLT t) {
        MOperand o; o.kind = Reg; o.vreg = v; o.type = t; o.rclass = reg_class_for(t); return o;
    }
    static MOperand Phys(int physreg, RegClass rc, LLT t) {
        MOperand o; o.kind = Reg; o.vreg = physreg; o.is_physical = true; o.rclass = rc; o.type = t; return o;
    }
    static MOperand ImmOp(int64_t v, LLT t = LLT::I64) { MOperand o; o.kind = Imm; o.imm = v; o.type = t; return o; }
    static MOperand FImmOp(double v, LLT t = LLT::F64) { MOperand o; o.kind = FImm; o.fimm = v; o.type = t; return o; }
    static MOperand Frame(int idx) { MOperand o; o.kind = FrameIdx; o.frame_idx = idx; o.type = LLT::Ptr; return o; }
    static MOperand Glob(std::string name) { MOperand o; o.kind = Global; o.global = std::move(name); o.type = LLT::Ptr; return o; }
    static MOperand Blk(MachineBasicBlock* b) { MOperand o; o.kind = Block; o.block = b; return o; }

    /// [base_vreg + disp], base is virtual until RegAlloc rewrites it.
    static MOperand MemReg(int base_vreg, LLT access_type, int64_t disp = 0) {
        MOperand o; o.kind = Mem; o.vreg = base_vreg; o.is_physical = false; o.type = access_type; o.imm = disp; o.frame_idx = -1;
        return o;
    }
    static MOperand MemPhys(int base_physreg, LLT access_type, int64_t disp = 0) {
        MOperand o; o.kind = Mem; o.vreg = base_physreg; o.is_physical = true; o.type = access_type; o.imm = disp; o.frame_idx = -1;
        return o;
    }
    /// [frame_objects[idx] + extra_disp], resolved to a concrete SP/FP
    /// displacement by PrologueEpilogInserter.
    static MOperand MemFrame(int idx, LLT access_type, int64_t extra_disp = 0) {
        MOperand o; o.kind = Mem; o.frame_idx = idx; o.type = access_type; o.imm = extra_disp; return o;
    }
};

/// One MIR instruction. At most one def (TensorC's current IR is single-def
/// SSA throughout, so multi-def instructions aren't needed).
///
/// Unlike LLVM's split between GlobalISel's generic MachineInstr and the
/// target-specific one produced by SelectionDAG, TensorC uses a single
/// MachineInstr type for both stages (this is actually closer to how LLVM's
/// MachineInstr *itself* works — one class, opcode meaning changes): before
/// TargetInstrInfo::select() runs, is_generic is true and `op` is valid; the
/// selector replaces each generic instruction with 1-N instructions that have
/// is_generic == false and `target_op` indexing that target's MCInstrDesc
/// table (see codegen/target/TargetInfo.h). Every later pass (Liveness,
/// RegAlloc, PrologueEpilogInserter, AsmPrinter) only ever sees target instrs.
struct MachineInstr {
    bool is_generic = true;
    MOp op = MOp::COPY;          // valid iff is_generic
    unsigned target_op = 0;      // valid iff !is_generic

    std::vector<MOperand> defs;
    std::vector<MOperand> uses;

    /// Valid iff op == ICMP || op == FCMP.
    CmpPred pred = CmpPred::IEQ;

    /// Direct-call target symbol (CALL only); kept separate from `uses` so
    /// call-argument operands stay a clean list.
    std::string callee;

    /// Physical registers this instruction clobbers beyond its defs (e.g. a
    /// CALL clobbers every caller-saved register per the target ABI). RegAlloc
    /// treats these as busy for the instruction's live-range slot.
    std::vector<int> clobbers_gpr;
    std::vector<int> clobbers_fpr;

    /// Valid iff !is_generic: the operand list in the exact order/spelling
    /// AsmPrinter should print it (e.g. AT&T's "src, dst" for x86 vs.
    /// RISC-V's "rd, rs1, rs2"). Kept separate from defs/uses because those
    /// two serve Liveness/RegAlloc (which need "what's read" vs. "what's
    /// written", not print order) — collapsing both roles into one list
    /// produced fragile order-inference bugs, so each target's select()
    /// fills defs/uses AND print_operands explicitly instead.
    ///
    /// Invariant each target's select() must uphold: every vreg that appears
    /// in print_operands must also appear (same vreg id) in defs or uses, so
    /// RegAlloc's operand rewrite — which walks all three vectors and
    /// replaces every occurrence of a given vreg id — keeps them consistent.
    std::vector<MOperand> print_operands;

    static MachineInstr make(MOp op) { MachineInstr mi; mi.is_generic = true; mi.op = op; return mi; }
    static MachineInstr makeTarget(unsigned target_op) {
        MachineInstr mi; mi.is_generic = false; mi.target_op = target_op; return mi;
    }
    MachineInstr& def(MOperand o) { defs.push_back(o); return *this; }
    MachineInstr& use(MOperand o) { uses.push_back(o); return *this; }
    MachineInstr& pr(MOperand o) { print_operands.push_back(o); return *this; }
};

struct MachineBasicBlock {
    std::string label;
    std::vector<MachineInstr> instrs;
    std::vector<MachineBasicBlock*> preds;
    std::vector<MachineBasicBlock*> succs;
};

struct VRegInfo {
    LLT type = LLT::I64;
};

/// A stack-frame slot: a source alloca, a spill slot, or an incoming
/// stack-passed argument. `offset` is resolved by PrologueEpilogInserter.
struct FrameObject {
    int64_t size = 8;
    int64_t align = 8;
    bool is_spill = false;
    bool is_incoming_arg = false;   // positive offset from FP (caller's stack, above saved-FP+return-addr)
    bool is_outgoing_arg = false;   // negative offset from FP, placed nearest SP (below spills/locals)
    int64_t offset = 0;
    bool resolved = false;
};

/// A target-independent function body. Blocks are heap-allocated so that
/// MachineBasicBlock* pointers stored in preds/succs/Block operands stay
/// valid as more blocks are appended.
struct MachineFunction {
    std::string name;
    std::vector<std::unique_ptr<MachineBasicBlock>> blocks;
    std::vector<VRegInfo> vregs;
    std::vector<FrameObject> frame_objects;
    std::vector<int> param_vregs;   // one vreg per incoming ir::Argument, in order
    LLT return_type = LLT::I64;
    bool has_return_value = false;

    /// Filled in by LinearScanRegAlloc, consumed by PrologueEpilogInserter:
    /// which callee-saved physical registers this function actually
    /// assigned to some vreg (only those need save/restore in the prologue).
    std::vector<int> used_callee_saved_gpr;
    std::vector<int> used_callee_saved_fpr;

    int new_vreg(LLT t) { vregs.push_back({t}); return static_cast<int>(vregs.size()) - 1; }
    LLT vreg_type(int v) const { return vregs[static_cast<size_t>(v)].type; }

    int new_frame_object(int64_t size, int64_t align = 8, bool is_spill = false) {
        FrameObject fo; fo.size = size; fo.align = align; fo.is_spill = is_spill;
        frame_objects.push_back(fo);
        return static_cast<int>(frame_objects.size()) - 1;
    }
    int new_incoming_arg_object(int64_t size, int64_t align = 8) {
        FrameObject fo; fo.size = size; fo.align = align; fo.is_incoming_arg = true;
        frame_objects.push_back(fo);
        return static_cast<int>(frame_objects.size()) - 1;
    }
    int new_outgoing_arg_object(int64_t size, int64_t align = 8) {
        FrameObject fo; fo.size = size; fo.align = align; fo.is_outgoing_arg = true;
        frame_objects.push_back(fo);
        return static_cast<int>(frame_objects.size()) - 1;
    }

    MachineBasicBlock* add_block(std::string label) {
        blocks.push_back(std::make_unique<MachineBasicBlock>());
        blocks.back()->label = std::move(label);
        return blocks.back().get();
    }
};

} // namespace codegen::mir
