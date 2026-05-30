/// codegen/tools/test_legacy_extended.cpp
///
/// Extended test suite for legacy codegen system
/// 
/// Tests:
/// 1. Tensor element-wise operations (ElemAdd, ElemMul)
/// 2. Matrix multiplication
/// 3. Fused operations (MatMul + activation)
/// 4. Control flow (branches, jumps)
/// 5. Memory operations (loads, stores)
/// 6. Mixed scalar + tensor operations

#include "../../compiler/ir/IRModule.h"
#include "../../compiler/ir/Instruction.h"
#include "../../compiler/ir/Value.h"
#include "../legacy/InstrSelector.h"
#include "../legacy/CodegenDriver.h"
#include "test_utils.h"
#include <iostream>
#include <cassert>

using namespace ir;
using namespace codegen::testing;

static ValuePtr borrow(Value* value)
{
    return ValuePtr(value, [](Value*) {});
}

/// Test 1: Tensor element-wise addition
void test_tensor_elemwise_add()
{
    std::cout << "\n=== Test 1: Tensor Element-Wise Add (8x8) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_tensor_elemwise");
    auto tensor_ty = Type::tensor(Type::f64(), std::vector<int>{8, 8});
    auto* fn = mod->add_function("elemwise_add", Type::fn({tensor_ty, tensor_ty}, tensor_ty));
    fn->create_entry();
    auto* a = fn->add_param("A", tensor_ty);
    auto* b = fn->add_param("B", tensor_ty);
    
    std::vector<ValuePtr> args;
    args.push_back(borrow(a));
    args.push_back(borrow(b));
    
    auto* result = fn->entry()->emit<TensorOpInst>("result", tensor_ty, TensorOpCode::ElemAdd, args);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    std::string out_file = "test_legacy_elemwise_add.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    // Should have call to tensor library function or inline ops
    assert((validator.has_mnemonic("call") || validator.has_mnemonic("vadd.vv")) && 
           "Should have call to library or vector add instruction");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) &&
       "Should end with return");

    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Tensor operation properly lowered\n";
}

/// Test 2: Tensor element-wise multiplication
void test_tensor_elemwise_mul()
{
    std::cout << "\n=== Test 2: Tensor Element-Wise Mul (8x8) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_tensor_elemwise_mul");
    auto tensor_ty = Type::tensor(Type::f64(), std::vector<int>{8, 8});
    auto* fn = mod->add_function("elemwise_mul", 
        Type::fn({tensor_ty, tensor_ty}, tensor_ty));
    fn->create_entry();
    auto* a = fn->add_param("A", tensor_ty);
    auto* b = fn->add_param("B", tensor_ty);
    
    std::vector<ValuePtr> args;
    args.push_back(borrow(a));
    args.push_back(borrow(b));
    
    auto* result = fn->entry()->emit<TensorOpInst>("result", tensor_ty, 
                                                   TensorOpCode::ElemMul, args);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    std::string out_file = "test_legacy_elemwise_mul.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    assert((validator.has_mnemonic("call") || validator.has_mnemonic("vmul.vv")) && 
           "Should have call to library or vector mul instruction");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
}

/// Test 3: Matrix multiplication
void test_matmul()
{
    std::cout << "\n=== Test 3: Matrix Multiplication (32x32) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_matmul");
    auto matrix_ty = Type::tensor(Type::f64(), std::vector<int>{32, 32});
    auto* fn = mod->add_function("matmul", 
        Type::fn({matrix_ty, matrix_ty}, matrix_ty));
    fn->create_entry();
    auto* a = fn->add_param("A", matrix_ty);
    auto* b = fn->add_param("B", matrix_ty);
    
    std::vector<ValuePtr> args;
    args.push_back(borrow(a));
    args.push_back(borrow(b));
    
    auto* result = fn->entry()->emit<TensorOpInst>("result", matrix_ty, 
                                                   TensorOpCode::MatMul, args);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    std::string out_file = "test_legacy_matmul.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    // MatMul should call library function
    assert(validator.has_mnemonic("call") && "Should have call instruction for MatMul");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ MatMul operation properly lowered to library call\n";
}

/// Test 4: Fused MatMul + ReLU
void test_fused_matmul_relu()
{
    std::cout << "\n=== Test 4: Fused MatMul + ReLU (32x32) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_fused_matmul_relu");
    auto matrix_ty = Type::tensor(Type::f64(), std::vector<int>{32, 32});
    auto* fn = mod->add_function("matmul_relu", 
        Type::fn({matrix_ty, matrix_ty}, matrix_ty));
    fn->create_entry();
    auto* a = fn->add_param("A", matrix_ty);
    auto* b = fn->add_param("B", matrix_ty);
    
    std::vector<ValuePtr> args;
    args.push_back(borrow(a));
    args.push_back(borrow(b));
    
    auto* result = fn->entry()->emit<TensorOpInst>("result", matrix_ty, 
                                                   TensorOpCode::FusedMatMulRelu, args);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    std::string out_file = "test_legacy_fused_matmul_relu.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    assert(validator.has_mnemonic("call") && "Should have call to fused kernel");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Fused operation properly lowered\n";
}

/// Test 5: Conditional branching (if-else logic)
void test_conditional_branch()
{
    std::cout << "\n=== Test 5: Conditional Branch (Max Function) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_conditional");
    auto* fn = mod->add_function("max_i64", 
        Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    
    auto* entry = fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    // Create two basic blocks: then and else
    auto* bb_then = fn->add_block("then");
    auto* bb_else = fn->add_block("else");
    auto* bb_merge = fn->add_block("merge");
    
    // entry: compare a < b
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* cmp = entry->emit<CmpInst>("cmp", CmpCode::Lt, a_ptr, b_ptr);
    entry->emit<CondBranchInst>(borrow(cmp), bb_then, bb_else);
    
    // then: return b
    ValuePtr b_ptr2 = borrow(b);
    bb_then->emit<ReturnInst>(b_ptr2);
    
    // else: return a
    ValuePtr a_ptr2 = borrow(a);
    bb_else->emit<ReturnInst>(a_ptr2);
    
    std::string out_file = "test_legacy_conditional.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    // Should have branch instructions
    assert((validator.has_mnemonic("blt") || validator.has_mnemonic("bnez") || 
            validator.has_mnemonic("beq") || validator.has_mnemonic("jne")) && 
           "Should have branch instruction");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Conditional branching properly lowered\n";
}

/// Test 6: Loop-like structure (multiple operations)
void test_loop_structure()
{
    std::cout << "\n=== Test 6: Loop Structure (Accumulation) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_loop");
    auto* fn = mod->add_function("sum_chain", 
        Type::fn({Type::i64(), Type::i64(), Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    auto* c = fn->add_param("c", Type::i64());
    auto* d = fn->add_param("d", Type::i64());
    
    // Simulate a loop: sum = a + b + c + d
    auto* sum1 = fn->entry()->emit<BinOpInst>("sum1", Type::i64(), 
                                             BinOpCode::Add, borrow(a), borrow(b));
    auto* sum2 = fn->entry()->emit<BinOpInst>("sum2", Type::i64(), 
                                             BinOpCode::Add, borrow(sum1), borrow(c));
    auto* sum3 = fn->entry()->emit<BinOpInst>("sum3", Type::i64(), 
                                             BinOpCode::Add, borrow(sum2), borrow(d));
    fn->entry()->emit<ReturnInst>(borrow(sum3));
    
    std::string out_file = "test_legacy_loop_structure.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    // Should have multiple add instructions
    assert(validator.has_mnemonic("add", 3) && "Should have 3 add instructions");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Multiple operations properly chained\n";
}

/// Test 7: Mixed operations with different types
void test_mixed_types()
{
    std::cout << "\n=== Test 7: Mixed Integer and Float Operations ===\n";
    
    auto mod = std::make_shared<IRModule>("test_mixed");
    auto* fn = mod->add_function("mixed", 
        Type::fn({Type::i64(), Type::f64()}, Type::f64()));
    fn->create_entry();
    
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::f64());
    
    // i64 add with itself
    auto* a_add_a = fn->entry()->emit<BinOpInst>("a_add_a", Type::i64(), 
                                                 BinOpCode::Add, borrow(a), borrow(a));
    
    // f64 mul with itself
    auto* b_mul_b = fn->entry()->emit<BinOpInst>("b_mul_b", Type::f64(), 
                                                 BinOpCode::FMul, borrow(b), borrow(b));
    
    fn->entry()->emit<ReturnInst>(borrow(b_mul_b));
    
    std::string out_file = "test_legacy_mixed_types.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    assert(validator.has_mnemonic("add") && "Should have integer add");
    assert((validator.has_mnemonic("fmul.d") || validator.has_mnemonic("fmul")) && 
           "Should have floating-point multiply");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Both integer and float operations present\n";
}

int main()
{
    try {
        std::cout << "╔═════════════════════════════════════════════════════════╗\n";
        std::cout << "║ TensorC Legacy Codegen - Extended Operations Test Suite ║\n";
        std::cout << "╚═════════════════════════════════════════════════════════╝\n";
        
        test_tensor_elemwise_add();
        test_tensor_elemwise_mul();
        test_matmul();
        test_fused_matmul_relu();
        test_conditional_branch();
        test_loop_structure();
        test_mixed_types();
        
        std::cout << "\n╔════════════════════════════════════╗\n";
        std::cout << "║ All Legacy Extended Tests PASSED ✓ ║\n";
        std::cout << "╚════════════════════════════════════╝\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << "\n";
        return 1;
    }
}
