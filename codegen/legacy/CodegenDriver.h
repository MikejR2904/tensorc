#pragma once
#include "../compiler/ir/IRModule.h"
#include <string>

namespace codegen {

// Lower an ir::Function to assembly text file
bool lower_function_to_asm(ir::Function* fn, const std::string& out_path);

} // namespace codegen
