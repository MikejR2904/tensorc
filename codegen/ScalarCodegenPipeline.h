#pragma once

/// codegen/ScalarCodegenPipeline.h
///
/// The new scalar/control-flow backend: GenericISel -> PhiElimination ->
/// CallLowering -> target instruction selection -> LinearScanRegAlloc ->
/// PrologueEpilogInserter -> AsmPrinter. Replaces
/// codegen/legacy/lower_function_to_asm's InstrSelector -> RegAlloc ->
/// AsmPrinter chain (see the backend redesign plan) for x86_64 and riscv64.
/// Tensor-op lowering (codegen/bridge, codegen/lowering, codegen/targets) is
/// untouched — CodegenDriver still splices its output in separately.

#include "../compiler/ir/IRModule.h"
#include "target/TargetInfo.h"
#include <memory>
#include <ostream>
#include <string>

namespace codegen {

class ScalarCodegenPipeline {
public:
    /// `target_name`: "x86_64" | "riscv64".
    explicit ScalarCodegenPipeline(const std::string& target_name);

    bool valid() const { return target_ != nullptr; }

    /// Lowers one function's scalar/control-flow IR to assembly text.
    /// Returns false (writing nothing) if the target name was unrecognized.
    bool lower_function(ir::Function& fn, std::ostream& out);

private:
    std::unique_ptr<target::Target> target_;
};

} // namespace codegen
