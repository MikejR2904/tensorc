#pragma once

namespace codegen {

struct Target {
    virtual ~Target() = default;
    virtual const char* name() const = 0;
};

struct RiscVTarget : Target {
    const char* name() const override { return "riscv64"; }
};

} // namespace codegen
