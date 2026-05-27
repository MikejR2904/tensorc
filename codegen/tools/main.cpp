#include "../../compiler/ir/IRModule.h"
#include "../../compiler/ir/Instruction.h"
#include "../../compiler/ir/Value.h"
#include "../InstrSelector.h"
#include "../CodegenDriver.h"
#include <iostream>
#include <fstream>

int main()
{
    using namespace ir;
    // Build IR function: fn add(a,b) { %c = add a,b; ret %c }
    auto mod = std::make_shared<IRModule>("<test>");
    auto* fn = mod->add_function("test_add", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    // Create shared_ptr wrappers using the constructor
    ValuePtr a_ptr(a);
    ValuePtr b_ptr(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::i64(), BinOpCode::Add, a_ptr, b_ptr);
    
    // Create ValuePtr for return
    ValuePtr c_ptr(c);
    fn->entry()->emit<ReturnInst>(c_ptr);

    // Lower to assembly
    std::string out_path = "out.s";
    if (!codegen::lower_function_to_asm(fn, out_path)) {
        std::cerr << "codegen failed\n";
        return 1;
    }
    std::cout << "Wrote " << out_path << "\n";
    
    // Print contents
    std::ifstream ifs(out_path);
    std::string line;
    std::cout << "=== Assembly output ===\n";
    while (std::getline(ifs, line)) std::cout << line << "\n";
    return 0;
}
