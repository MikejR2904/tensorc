#pragma once

#include "../TargetInfo.h"
#include <memory>

namespace codegen::target::riscv64 {

std::unique_ptr<Target> create_riscv64_target();

} // namespace codegen::target::riscv64
