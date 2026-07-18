#pragma once

/// codegen/target/x86_64/X86Target.h
///
/// Bundles the x86-64 System V backend into the codegen::target::Target
/// interface. This is the only file outside this directory that needs to
/// know x86-64 exists (referenced once, from codegen/target/TargetInfo.cpp's
/// create_target() factory).

#include "../TargetInfo.h"
#include <memory>

namespace codegen::target::x86_64 {

std::unique_ptr<Target> create_x86_64_target();

} // namespace codegen::target::x86_64
