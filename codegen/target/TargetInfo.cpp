#include "TargetInfo.h"
#include "x86_64/X86Target.h"
#include "riscv64/RiscVTarget.h"

namespace codegen::target {

std::unique_ptr<Target> create_target(const std::string& name) {
    if (name == "x86_64") return x86_64::create_x86_64_target();
    if (name == "riscv64") return riscv64::create_riscv64_target();
    return nullptr;
}

} // namespace codegen::target
