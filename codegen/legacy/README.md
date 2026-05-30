## Overview

The legacy system implements a basic RISC-V 64-bit code generation pipeline for simple operations:

```
IR::Instruction → MachineInstr → PhysicalRegs → RV64 Assembly
```

## Components

### MachineInstr.h & MachineFunction.h
Low-level IR representation:
- **MachineOperand**: Represents operands (registers, immediates, memory)
  - Physical/virtual registers, immediate values, labels, memory addresses
  - Register classes: GPR, FPR, Vector
- **MachineInstr**: Represents a machine-level instruction
  - Opcode + operand list
  - No semantics checking (assumes validity from higher levels)
- **MachineBasicBlock**: Sequence of machine instructions with a label
- **MachineFunction**: Collection of blocks forming a function

### InstrSelector (IR → MachineInstr)
Converts IR Instructions to MachineInstr sequences via visitor pattern.

**Supported Operations**:
- **Scalar Ops**: Add, Sub, Mul, Div (int & float)
- **Memory**: Load, Store (1-64 bit values)
- **Control Flow**: Branch, CondBranch (using blt/bge)
- **Function**: Call, Return
- **Pseudo-Ops**: Tensor operations (calls to library functions)

**Output**:
- All operands use virtual registers (vregs)
- VRegs are stable small integers (not pointer-truncated)

### RegAlloc (Vreg → Preg)
Two-pass greedy register allocator.

**Register Banks**:
- GPR: x10-x17 (a0-a7) — 8 registers
- FPR: f10-f17 (fa0-fa7) — 8 registers
- Vector: v8-v23 — 16 registers

**Strategy**:
1. **Pass 1**: Scan all instructions; record vreg → RegClass mapping
2. **Pass 2**: Walk again; assign physical registers or spill to stack
   - Spilled operands annotated with slot number
   - Load/store instructions inserted around uses/defs

**Limitations**:
- No liveness analysis (treats all vregs as live from first to last use)
- No register coalescing
- Simple greedy allocation (no graph coloring)

### AsmPrinter (MachineInstr → RV64 Assembly)
Emits GNU assembler (GAS) syntax for RISC-V 64-bit.

**Features**:
- ABI name aliases (x10→a0, x1→ra, etc.)
- Physical register resolution after RegAlloc
- GAS directive output (.text, .global, .align)

**Output Format**:
```gas
.text
.global function_name
function_name:
  block_label:
    opcode    op0, op1, ...
    ...
```

### CodegenDriver
Orchestrates the legacy pipeline.

**Entry Point**: `lower_function_to_asm(ir::Function* fn, const std::string& out_path)`

**Flow**:
1. Create MachineFunction from IR::Function
2. Run InstrSelector visitor on all IR instructions
3. Run RegAlloc to map vregs → pregs
4. Run AsmPrinter to emit assembly text

## Usage Example

```cpp
#include "legacy/CodegenDriver.h"
#include "compiler/ir/IRModule.h"

// Build an IR function...
auto mod = std::make_shared<ir::IRModule>("mymodule");
auto* fn = mod->add_function("my_add", ir::Type::fn({ir::Type::i64(), ir::Type::i64()}, ir::Type::i64()));
// ... populate fn with IR instructions ...

// Lower to assembly
bool success = codegen::lower_function_to_asm(fn, "output.s");
```

## Limitations & Future Work

1. **Basic Register Allocation**: No live-range analysis or spilling heuristics
2. **No Optimization**: No peephole, scheduling, or common subexpression elimination
3. **RISC-V Only**: Hard-coded for RV64GV (no x86, ARM support)
4. **Scalar Focus**: Designed for simple operations, not vectorization
5. **No Calling Conventions**: Simplistic parameter/return value handling

## See Also

- **../lowering/**: Progressive lowering for tensor operations (recommended for new code)
- **../targets/**: Target abstraction layer used by the new system
- **../PROGRESSIVE_LOWERING.md**: Architecture guide for the recommended approach
