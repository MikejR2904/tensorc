#pragma once

/// codegen/mir/CallLowering.h
///
/// Assigns real ABI locations to incoming parameters, outgoing call
/// arguments, and return values — the layer that turns the old
/// codegen/legacy/InstrSelector's hardcoded "first 8 args in a0-a7, no
/// stack-passed-argument support, no callee-saved handling" into real,
/// per-target, register-count-agnostic argument passing (System V AMD64 /
/// LP64D, or any future target's CallingConv). Runs on generic MIR, after
/// GenericISel + PhiElimination and before TargetInstrInfo::select() — by
/// the time selection sees a CALL or RET, all ABI decisions are already
/// expressed as plain COPY/LOAD/STORE instructions to/from physical
/// registers or frame slots, so select() has nothing target-convention-aware
/// left to do.
///
/// Entirely target-independent *code*, parameterized by target *data*
/// (CallingConv + TargetRegisterInfo) — no architecture branches live here.

#include "MachineIR.h"
#include "../target/TargetInfo.h"

namespace codegen::mir {

void lower_calls(MachineFunction& mf, const target::CallingConv& cc, const target::TargetRegisterInfo& tri);

} // namespace codegen::mir
