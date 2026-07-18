#pragma once

/// codegen/mir/PhiElimination.h
///
/// Destroys SSA form. The old codegen/legacy/InstrSelector lowered PHI to a
/// `mv` emitted in the phi's *own* block, which its own comment admits is
/// only "a conservative approximation" — it's unsound in general (the
/// classic lost-copy / swap problem on critical edges and parallel phi
/// copies). This pass does it properly: every PHI is replaced by copies
/// inserted at the end of each predecessor block, critical edges are split
/// first so a copy never runs on a path shared with unrelated control flow,
/// and each predecessor's copies are sequenced through fresh temporaries so
/// swapping phis (dst_a fed by dst_b's old value and vice versa) can never
/// clobber a source before it's read.

#include "MachineIR.h"

namespace codegen::mir {

/// Must run on generic MIR, after GenericISel and before
/// TargetInstrInfo::select() (selection assumes a PHI-free instruction stream).
void eliminate_phis(MachineFunction& mf);

} // namespace codegen::mir
