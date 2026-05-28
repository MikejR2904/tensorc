#include "../../compiler/ir/IRModule.h"
#include "../../compiler/ir/Instruction.h"
#include "../../compiler/ir/Value.h"
#include "../legacy/InstrSelector.h"
#include "../legacy/CodegenDriver.h"
#include <iostream>
#include <fstream>
#include <cassert>

using namespace ir;

static ValuePtr borrow(Value* value)
{
    return ValuePtr(value, [](Value*) {});
}

// Test 1: Simple scalar add
void test_simple_add()
{
    std::cout << "\n=== Test 1: Simple Add (Scalar) ===\n";
    auto mod = std::make_shared<IRModule>("<test_add>");
    auto* fn = mod->add_function("add_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::i64(), BinOpCode::Add, a_ptr, b_ptr);
    fn->entry()->emit<ReturnInst>(borrow(c));
    
    codegen::lower_function_to_asm(fn, "test_1_add.s");
    
    // Print assembly
    std::ifstream ifs("test_1_add.s");
    std::string line;
    while (std::getline(ifs, line)) std::cout << line << "\n";
}

// Test 2: Scalar multiply
void test_simple_mul()
{
    std::cout << "\n=== Test 2: Simple Multiply (Scalar) ===\n";
    auto mod = std::make_shared<IRModule>("<test_mul>");
    auto* fn = mod->add_function("mul_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::i64(), BinOpCode::Mul, a_ptr, b_ptr);
    fn->entry()->emit<ReturnInst>(borrow(c));
    
    codegen::lower_function_to_asm(fn, "test_2_mul.s");
    
    std::ifstream ifs("test_2_mul.s");
    std::string line;
    while (std::getline(ifs, line)) std::cout << line << "\n";
}

// Test 3: Tensor element-wise add (intermediate)
void test_tensor_elem_add()
{
    std::cout << "\n=== Test 3: Tensor ElemAdd (Intermediate) ===\n";
    auto mod = std::make_shared<IRModule>("<test_elem_add>");
    
    // Build tensor type (e.g., Tensor[f64, {8, 8}])
    auto tensor_ty = Type::tensor(Type::f64(), std::vector<int>{8, 8});
    auto* fn = mod->add_function("elem_add_f64", Type::fn({tensor_ty, tensor_ty}, tensor_ty));
    fn->create_entry();
    auto* a = fn->add_param("a", tensor_ty);
    auto* b = fn->add_param("b", tensor_ty);
    
    // Create vector of ValuePtr for tensor op arguments
    std::vector<ValuePtr> args;
    args.push_back(borrow(a));
    args.push_back(borrow(b));
    
    auto* result = fn->entry()->emit<TensorOpInst>("result", tensor_ty, TensorOpCode::ElemAdd, args);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    codegen::lower_function_to_asm(fn, "test_3_elem_add.s");
    
    std::ifstream ifs("test_3_elem_add.s");
    std::string line;
    while (std::getline(ifs, line)) std::cout << line << "\n";
}

// Test 4: Matrix multiply (intermediate-complex)
void test_matmul()
{
    std::cout << "\n=== Test 4: MatMul (Complex) ===\n";
    auto mod = std::make_shared<IRModule>("<test_matmul>");
    
    auto matrix_ty = Type::tensor(Type::f64(), std::vector<int>{64, 64});
    auto* fn = mod->add_function("matmul_f64", Type::fn({matrix_ty, matrix_ty}, matrix_ty));
    fn->create_entry();
    auto* a = fn->add_param("A", matrix_ty);
    auto* b = fn->add_param("B", matrix_ty);
    
    std::vector<ValuePtr> args;
    args.push_back(borrow(a));
    args.push_back(borrow(b));
    
    auto* result = fn->entry()->emit<TensorOpInst>("C", matrix_ty, TensorOpCode::MatMul, args);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    codegen::lower_function_to_asm(fn, "test_4_matmul.s");
    
    std::ifstream ifs("test_4_matmul.s");
    std::string line;
    while (std::getline(ifs, line)) std::cout << line << "\n";
}

// Test 5: Fused MatMul + ReLU (complex with kernel fusion)
void test_fused_matmul_relu()
{
    std::cout << "\n=== Test 5: FusedMatMulReLU (Kernel Fusion) ===\n";
    auto mod = std::make_shared<IRModule>("<test_fused>");
    
    auto matrix_ty = Type::tensor(Type::f64(), std::vector<int>{32, 32});
    auto* fn = mod->add_function("matmul_relu_f64", Type::fn({matrix_ty, matrix_ty}, matrix_ty));
    fn->create_entry();
    auto* a = fn->add_param("A", matrix_ty);
    auto* b = fn->add_param("B", matrix_ty);
    
    std::vector<ValuePtr> args;
    args.push_back(borrow(a));
    args.push_back(borrow(b));
    
    auto* result = fn->entry()->emit<TensorOpInst>("C", matrix_ty, TensorOpCode::FusedMatMulRelu, args);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    codegen::lower_function_to_asm(fn, "test_5_fused.s");
    
    std::ifstream ifs("test_5_fused.s");
    std::string line;
    while (std::getline(ifs, line)) std::cout << line << "\n";
}

// Test 6: Complex chain with branches (control flow)
void test_branching()
{
    std::cout << "\n=== Test 6: Branching (Control Flow) ===\n";
    auto mod = std::make_shared<IRModule>("<test_branch>");
    
    auto* fn = mod->add_function("max_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    auto* entry = fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    // Create basic blocks: true_block and false_block
    auto* true_bb = fn->add_block("true_block");
    auto* false_bb = fn->add_block("false_block");
    auto* merge_bb = fn->add_block("merge_block");
    
    // Entry: compare a < b
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* cmp = entry->emit<CmpInst>("cmp", CmpCode::Lt, a_ptr, b_ptr);
    entry->emit<CondBranchInst>(borrow(cmp), true_bb, false_bb);
    
    // true_block: return b
    true_bb->emit<ReturnInst>(b_ptr);
    
    // false_block: return a
    false_bb->emit<ReturnInst>(a_ptr);
    
    codegen::lower_function_to_asm(fn, "test_6_branch.s");
    
    std::ifstream ifs("test_6_branch.s");
    std::string line;
    while (std::getline(ifs, line)) std::cout << line << "\n";
}

int main()
{
    std::cout << "=== TensorC Codegen Test Suite ===\n";
    
    try {
        test_simple_add();
        test_simple_mul();
        test_tensor_elem_add();
        test_matmul();
        test_fused_matmul_relu();
        test_branching();
        
        std::cout << "\n=== All tests completed successfully! ===\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
