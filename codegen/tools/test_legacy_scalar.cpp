/// codegen/tools/test_legacy_scalar.cpp
///
/// Test suite for legacy codegen system - scalar operations
/// 
/// This tests the basic IR → MachineInstr → RegAlloc → Assembly pipeline
/// for scalar integer and floating-point operations.
///
/// Tests:
/// 1. Integer addition (add i64)
/// 2. Integer multiplication (mul i64)
/// 3. Floating-point operations (fadd, fmul)
/// 4. Division operations (div i64, fdiv f64)
/// 5. Mixed operations in sequence

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

/// Test 1: Integer addition
void test_i64_add()
{
    std::cout << "\n=== Test 1: Integer Addition (i64) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_i64_add");
    auto* fn = mod->add_function("add_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::i64(), BinOpCode::Add, a_ptr, b_ptr);
    fn->entry()->emit<ReturnInst>(borrow(c));
    
    // Generate assembly
    std::string out_file = "test_legacy_i64_add.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    // Validate output
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    // Should contain add instruction
    assert(validator.has_mnemonic("add") && "Should have 'add' instruction");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Assembly contains 'add' instruction\n";
    std::cout << "✓ Properly ends with return\n";
}

/// Test 2: Integer multiplication
void test_i64_mul()
{
    std::cout << "\n=== Test 2: Integer Multiplication (i64) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_i64_mul");
    auto* fn = mod->add_function("mul_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::i64(), BinOpCode::Mul, a_ptr, b_ptr);
    fn->entry()->emit<ReturnInst>(borrow(c));
    
    std::string out_file = "test_legacy_i64_mul.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    assert(validator.has_mnemonic("mul") && "Should have 'mul' instruction");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Assembly contains 'mul' instruction\n";
}

/// Test 3: Floating-point addition
void test_f64_add()
{
    std::cout << "\n=== Test 3: Floating-Point Addition (f64) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_f64_add");
    auto* fn = mod->add_function("fadd_f64", Type::fn({Type::f64(), Type::f64()}, Type::f64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::f64());
    auto* b = fn->add_param("b", Type::f64());
    
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::f64(), BinOpCode::FAdd, a_ptr, b_ptr);
    fn->entry()->emit<ReturnInst>(borrow(c));
    
    std::string out_file = "test_legacy_f64_add.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    // Should have floating-point add
    assert((validator.has_mnemonic("fadd.d") || validator.has_mnemonic("fadd")) && 
           "Should have floating-point add instruction");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Assembly contains floating-point add\n";
}

/// Test 4: Integer division
void test_i64_div()
{
    std::cout << "\n=== Test 4: Integer Division (i64) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_i64_div");
    auto* fn = mod->add_function("div_i64", Type::fn({Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::i64(), BinOpCode::Div, a_ptr, b_ptr);
    fn->entry()->emit<ReturnInst>(borrow(c));
    
    std::string out_file = "test_legacy_i64_div.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    assert((validator.has_mnemonic("div") || validator.has_mnemonic("udiv")) && 
           "Should have division instruction");
   assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Assembly contains division instruction\n";
}

/// Test 5: Floating-point multiplication
void test_f64_mul()
{
    std::cout << "\n=== Test 5: Floating-Point Multiplication (f64) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_f64_mul");
    auto* fn = mod->add_function("fmul_f64", Type::fn({Type::f64(), Type::f64()}, Type::f64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::f64());
    auto* b = fn->add_param("b", Type::f64());
    
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* c = fn->entry()->emit<BinOpInst>("c", Type::f64(), BinOpCode::FMul, a_ptr, b_ptr);
    fn->entry()->emit<ReturnInst>(borrow(c));
    
    std::string out_file = "test_legacy_f64_mul.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    assert((validator.has_mnemonic("fmul.d") || validator.has_mnemonic("fmul")) && 
           "Should have floating-point multiply");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Assembly contains floating-point multiply\n";
}

/// Test 6: Chained operations (Add then Mul)
void test_chained_ops()
{
    std::cout << "\n=== Test 6: Chained Operations (Add + Mul) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_chained");
    auto* fn = mod->add_function("chained", Type::fn({Type::i64(), Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    auto* c = fn->add_param("c", Type::i64());
    
    // t1 = a + b
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* t1 = fn->entry()->emit<BinOpInst>("t1", Type::i64(), BinOpCode::Add, a_ptr, b_ptr);
    
    // result = t1 * c
    ValuePtr t1_ptr = borrow(t1);
    ValuePtr c_ptr = borrow(c);
    auto* result = fn->entry()->emit<BinOpInst>("result", Type::i64(), BinOpCode::Mul, t1_ptr, c_ptr);
    fn->entry()->emit<ReturnInst>(borrow(result));
    
    std::string out_file = "test_legacy_chained.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    assert(validator.has_mnemonic("add") && "Should have 'add' instruction");
    assert(validator.has_mnemonic("mul") && "Should have 'mul' instruction");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions\n";
    std::cout << "✓ Contains both 'add' and 'mul' instructions\n";
}

/// Test 7: Many operands (stress test for register allocation)
void test_register_pressure()
{
    std::cout << "\n=== Test 7: Register Pressure (Many Operations) ===\n";
    
    auto mod = std::make_shared<IRModule>("test_reg_pressure");
    auto* fn = mod->add_function("complex", 
        Type::fn({Type::i64(), Type::i64(), Type::i64(), Type::i64()}, Type::i64()));
    fn->create_entry();
    auto* a = fn->add_param("a", Type::i64());
    auto* b = fn->add_param("b", Type::i64());
    auto* c = fn->add_param("c", Type::i64());
    auto* d = fn->add_param("d", Type::i64());
    
    // Build: ((a + b) * (c + d)) + ((a * b) + (c * d))
    ValuePtr a_ptr = borrow(a);
    ValuePtr b_ptr = borrow(b);
    auto* ab = fn->entry()->emit<BinOpInst>("ab", Type::i64(), BinOpCode::Add, a_ptr, b_ptr);
    
    ValuePtr c_ptr = borrow(c);
    ValuePtr d_ptr = borrow(d);
    auto* cd = fn->entry()->emit<BinOpInst>("cd", Type::i64(), BinOpCode::Add, c_ptr, d_ptr);
    
    ValuePtr ab_ptr = borrow(ab);
    ValuePtr cd_ptr = borrow(cd);
    auto* ab_x_cd = fn->entry()->emit<BinOpInst>("ab_x_cd", Type::i64(), BinOpCode::Mul, ab_ptr, cd_ptr);
    
    a_ptr = borrow(a);
    b_ptr = borrow(b);
    auto* a_x_b = fn->entry()->emit<BinOpInst>("a_x_b", Type::i64(), BinOpCode::Mul, a_ptr, b_ptr);
    
    c_ptr = borrow(c);
    d_ptr = borrow(d);
    auto* c_x_d = fn->entry()->emit<BinOpInst>("c_x_d", Type::i64(), BinOpCode::Mul, c_ptr, d_ptr);
    
    auto* axb_plus_cxd = fn->entry()->emit<BinOpInst>("axb_plus_cxd", Type::i64(), 
                                                       BinOpCode::Add, borrow(a_x_b), borrow(c_x_d));
    
    auto* final = fn->entry()->emit<BinOpInst>("final", Type::i64(), BinOpCode::Add, 
                                               borrow(ab_x_cd), borrow(axb_plus_cxd));
    fn->entry()->emit<ReturnInst>(borrow(final));
    
    std::string out_file = "test_legacy_reg_pressure.s";
    bool success = codegen::lower_function_to_asm(fn, out_file);
    assert(success && "Codegen should succeed even under register pressure");
    
    std::string asm_text = read_asm_file(out_file);
    AssemblyValidator validator(asm_text);
    
    // Should have multiple operations
    assert(validator.instr_count() >= 6 && "Should generate multiple instructions");
    assert((validator.ends_with_return() || validator.has_mnemonic("ret")) && "Should end with return");
    
    std::cout << "✓ Generated " << validator.instr_count() << " instructions under register pressure\n";
    std::cout << "✓ Successfully handled multiple temporaries\n";
}

int main()
{
    try {
        std::cout << "╔═══════════════════════════════════════════════════════╗\n";
        std::cout << "║ TensorC Legacy Codegen - Scalar Operations Test Suite ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════╝\n";
        
        test_i64_add();
        test_i64_mul();
        test_f64_add();
        test_i64_div();
        test_f64_mul();
        test_chained_ops();
        test_register_pressure();
        
        std::cout << "\n╔════════════════════════════════════╗\n";
        std::cout << "║ All Legacy Scalar Tests PASSED ✓   ║\n";
        std::cout << "╚════════════════════════════════════╝\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << "\n";
        return 1;
    }
}
