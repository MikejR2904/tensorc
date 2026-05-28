/// codegen/targets/TargetEmitter.cpp

#include "TargetEmitter.h"
#include "X86TargetEmitter.h"
#include "RiscVTargetEmitter.h"
#include <stdexcept>
#include <sstream>

namespace codegen::targets {

std::unique_ptr<TargetEmitter> create_target_emitter(const std::string& target)
{
    if (target == "x86_64" || target == "x86") {
        return std::make_unique<X86TargetEmitter>();
    } else if (target == "riscv64" || target == "riscv") {
        return std::make_unique<RiscVTargetEmitter>();
    } else if (target == "arm64" || target == "aarch64") {
        // TODO: ARM64 emitter
        std::ostringstream oss;
        oss << "Target '" << target << "' not yet implemented";
        throw std::runtime_error(oss.str());
    } else {
        std::ostringstream oss;
        oss << "Unknown target: '" << target << "'";
        throw std::runtime_error(oss.str());
    }
}

} // namespace codegen::targets
