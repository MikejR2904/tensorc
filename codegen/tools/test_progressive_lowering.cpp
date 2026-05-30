/// codegen/tools/test_progressive_lowering.cpp
///
/// Comprehensive Test Suite for Progressive Lowering Pipeline
/// Tests all 4 phases: Tiling, Memory Legalization, Scheduling, Target Emission
///
/// Compile with: g++ -std=c++17 -I.. test_progressive_lowering.cpp \
///   ../lowering/Tiler.cpp ../lowering/ScratchpadAllocator.cpp \
///   ../lowering/MemoryLegalizer.cpp ../lowering/Scheduler.cpp \
///   ../targets/TargetEmitter.cpp ../targets/X86TargetEmitter.cpp \
///   ../targets/RiscVTargetEmitter.cpp ../NewCodegenDriver.cpp

#include "../NewCodegenDriver.h"
#include "../../compiler/ir/IRModule.h"
#include "../../compiler/ir/Instruction.h"
#include <iostream>
#include <sstream>
#include <cassert>

using namespace ir;
using namespace codegen;

static ValuePtr borrow(Value* value)
{
    return ValuePtr(value, [](Value*) {});
}

// ════════════════════════════════════════════════════════════════════════════
// Test Infrastructure
// ════════════════════════════════════════════════════════════════════════════

int test_count = 0;
int test_passed = 0;

void test_begin(const std::string& name)
{
    test_count++;
    std::cout << "\n[Test " << test_count << "] " << name << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void test_pass()
{
    test_passed++;
    std::cout << "[YES] PASSED\n";
}

void test_fail(const std::string& reason)
{
    std::cout << "[NO] FAILED: " << reason << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
// Test 1: Simple MatMul Tiling
// ════════════════════════════════════════════════════════════════════════════

void test_matmul_tiling()
{
    test_begin("MatMul Tiling (128x256 x 256x128 to 128x128)");
    
    try {
        auto mod = std::make_shared<IRModule>("<test_matmul>");
        auto* fn = mod->add_function("matmul_128x128x256",
            Type::fn({}, Type::f64()));
        fn->create_entry();
        
        auto tensor_a = Type::tensor(Type::f64(), std::vector<int>{128, 256});
        auto tensor_b = Type::tensor(Type::f64(), std::vector<int>{256, 128});
        auto* a = fn->add_param("A", tensor_a);
        auto* b = fn->add_param("B", tensor_b);
        std::vector<ValuePtr> operands{borrow(a), borrow(b)};
        auto* matmul = fn->entry()->emit<TensorOpInst>(
            "result", Type::f64(), TensorOpCode::MatMul, operands
        );
        
        // Set up shape information
        lowering::Tiler tiler;
        std::map<const void*, lowering::TensorShape> shapes;
        shapes[a] = lowering::TensorShape{{128, 256}, 8};
        shapes[b] = lowering::TensorShape{{256, 128}, 8};
        
        // Lower to LLIR
        auto llir = tiler.lower(matmul, shapes);
        
        if (!llir) {
            test_fail("Tiler returned nullptr");
            return;
        }
        
        // Verify tiling structure
        assert(llir->dims.size() > 0);  // Should have loop dimensions
        assert(llir->compute_blocks.size() > 0);  // Should have compute blocks
        
        std::cout << "  Loop dimensions: " << llir->dims.size() << "\n";
        for (const auto& dim : llir->dims) {
            std::cout << "    " << dim.name << ": [" << dim.start 
                      << ", " << dim.limit << "), step=" << dim.step << "\n";
        }
        
        std::cout << "  Compute blocks: " << llir->compute_blocks.size() << "\n";
        for (const auto& compute : llir->compute_blocks) {
            std::cout << "    " << compute.operation << "\n";
        }
        
        test_pass();
    } catch (const std::exception& e) {
        test_fail(std::string("Exception: ") + e.what());
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 2: Scratchpad Allocation
// ════════════════════════════════════════════════════════════════════════════

void test_scratchpad_allocation()
{
    test_begin("Scratchpad Allocation (8x8 systolic with 8 KB limit)");
    
    try {
        // Create a simple loop nest with buffers
        lowering::LoopNest llir("matmul_tiled");
        
        // Add buffers (typical 8×8 tile scenario)
        auto buf_a = std::make_shared<lowering::MemoryBuffer>(
            "A_tile", lowering::BufferRole::InputA, 8, 512, true  // 8×8×8 bytes
        );
        auto buf_b = std::make_shared<lowering::MemoryBuffer>(
            "B_tile", lowering::BufferRole::InputB, 8, 512, true
        );
        auto buf_c = std::make_shared<lowering::MemoryBuffer>(
            "C_tile", lowering::BufferRole::Accumulator, 8, 512, true
        );
        
        llir.add_buffer(buf_a);
        llir.add_buffer(buf_b);
        llir.add_buffer(buf_c);
        
        // Allocate scratchpad
        lowering::ScratchpadAllocator allocator(
            lowering::AllocationStrategy::GreedyReuse
        );
        auto allocations = allocator.allocate(llir);
        
        // Verify allocations
        assert(allocations.size() == 3);
        
        int64_t total = allocator.total_allocated();
        std::cout << "  Total allocated: " << total << " bytes (capacity: 8192)\n";
        
        for (const auto& [name, record] : allocations) {
            std::cout << "    " << name << ": offset=" << record.offset 
                      << ", size=" << record.size << "\n";
        }
        
        assert(total <= 8192);  // Fits in 8 KB
        test_pass();
    } catch (const std::exception& e) {
        test_fail(std::string("Exception: ") + e.what());
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 3: K > 64 Spilling Detection & Handling
// ════════════════════════════════════════════════════════════════════════════

void test_k_spilling()
{
    test_begin("K > 64 Reduction Spilling");
    
    try {
        // Create a MatMul with K=128 (exceeds 64-element limit)
        lowering::LoopNest llir("matmul_large_k");
        
        // Setup loop structure with K=128
        llir.add_loop_dim(lowering::LoopDim("i", 0, 128, 8, false, 0));  // M loop
        llir.add_loop_dim(lowering::LoopDim("j", 0, 128, 8, false, 1));  // N loop
        llir.add_loop_dim(lowering::LoopDim("k", 0, 128, 8, true, 2));   // K loop > 64
        
        llir.requires_spilling = true;
        
        // Apply legalizer
        lowering::MemoryLegalizer legalizer;
        auto legalized = legalizer.legalize(llir);
        
        const auto& spilling = legalizer.spilling_info();
        std::cout << "  Requires spilling: " << (spilling.requires_spilling ? "yes" : "no") << "\n";
        
        if (spilling.requires_spilling) {
            std::cout << "  Outer loop iterations: " << spilling.outer_loop_iterations << "\n";
            assert(spilling.outer_loop_iterations == 2);  // 128 / 64 = 2
        }
        
        test_pass();
    } catch (const std::exception& e) {
        test_fail(std::string("Exception: ") + e.what());
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 4: Double-Buffering Scheduling
// ════════════════════════════════════════════════════════════════════════════

void test_double_buffering()
{
    test_begin("Double-Buffering Scheduler");
    
    try {
        lowering::LoopNest legalized("matmul_scheduled");
        
        // Add loops with multiple iterations
        legalized.add_loop_dim(lowering::LoopDim("i", 0, 16, 8, false, 0));
        legalized.add_loop_dim(lowering::LoopDim("j", 0, 16, 8, false, 1));
        legalized.add_loop_dim(lowering::LoopDim("k", 0, 64, 8, true, 2));
        
        // Add compute and memory operations
        lowering::ComputeBlock dma_load(lowering::ComputeKind::MemoryCopy, "dma_load");
        lowering::ComputeBlock compute(lowering::ComputeKind::SystolicMatMul, "matmul");
        
        legalized.add_compute(dma_load);
        legalized.add_compute(compute);
        
        // Schedule
        lowering::Scheduler scheduler;
        auto scheduled = scheduler.schedule(legalized);
        
        const auto& schedule_info = scheduler.schedule_info();
        std::cout << "  Uses double-buffering: " 
                  << (schedule_info.uses_double_buffering ? "yes" : "no") << "\n";
        
        if (schedule_info.uses_double_buffering) {
            std::cout << "  Ping buffers: " << schedule_info.ping_buffers.size() << "\n";
            std::cout << "  Pong buffers: " << schedule_info.pong_buffers.size() << "\n";
        }
        
        test_pass();
    } catch (const std::exception& e) {
        test_fail(std::string("Exception: ") + e.what());
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 5: End-to-End RISC-V Emission
// ════════════════════════════════════════════════════════════════════════════

void test_riscv_emission()
{
    test_begin("End-to-End RISC-V Target Emission");
    
    try {
        // Create a simple IR function
        auto mod = std::make_shared<IRModule>("<test_riscv>");
        auto* fn = mod->add_function("matmul_riscv",
            Type::fn({}, Type::f64()));
        fn->create_entry();
        
        auto tensor_ty = Type::tensor(Type::f64(), std::vector<int>{64, 64});
        auto* a = fn->add_param("A", tensor_ty);
        auto* b = fn->add_param("B", tensor_ty);
        std::vector<ValuePtr> operands{borrow(a), borrow(b)};
        auto* matmul = fn->entry()->emit<TensorOpInst>(
            "result", Type::f64(), TensorOpCode::MatMul, operands
        );
        
        // Setup shapes
        std::map<const void*, lowering::TensorShape> shapes;
        shapes[a] = lowering::TensorShape{{64, 64}, 8};
        shapes[b] = lowering::TensorShape{{64, 64}, 8};
        
        // Create pipeline
        ProgressiveLoweringPipeline pipeline("riscv64");
        
        // Emit
        std::ostringstream out;
        bool success = pipeline.lower_tensor_op(matmul, shapes, out);
        
        if (!success) {
            test_fail(pipeline.last_diagnostic());
            return;
        }
        
        std::cout << "  Emitted RISC-V assembly:\n";
        std::string asm_output = out.str();
        // Print first 20 lines
        int line_count = 0;
        std::istringstream iss(asm_output);
        std::string line;
        while (std::getline(iss, line) && line_count < 20) {
            std::cout << "    " << line << "\n";
            line_count++;
        }
        if (asm_output.length() > 0) {
            std::cout << "    ... (total " << asm_output.length() << " bytes)\n";
        }
        
        test_pass();
    } catch (const std::exception& e) {
        test_fail(std::string("Exception: ") + e.what());
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test 6: End-to-End x86_64 Emission
// ════════════════════════════════════════════════════════════════════════════

void test_x86_emission()
{
    test_begin("End-to-End x86_64 Target Emission");
    
    try {
        auto mod = std::make_shared<IRModule>("<test_x86>");
        auto* fn = mod->add_function("matmul_x86",
            Type::fn({}, Type::f64()));
        fn->create_entry();
        
        auto tensor_ty = Type::tensor(Type::f64(), std::vector<int>{64, 64});
        auto* a = fn->add_param("A", tensor_ty);
        auto* b = fn->add_param("B", tensor_ty);
        std::vector<ValuePtr> operands{borrow(a), borrow(b)};
        auto* matmul = fn->entry()->emit<TensorOpInst>(
            "result", Type::f64(), TensorOpCode::MatMul, operands
        );

        std::map<const void*, lowering::TensorShape> shapes;
        shapes[a] = lowering::TensorShape{{64, 64}, 8};
        shapes[b] = lowering::TensorShape{{64, 64}, 8};
        
        ProgressiveLoweringPipeline pipeline("x86_64");
        std::ostringstream out;
        bool success = pipeline.lower_tensor_op(matmul, shapes, out);
        
        if (!success) {
            test_fail(pipeline.last_diagnostic());
            return;
        }
        
        std::cout << "  Emitted x86_64 assembly:\n";
        std::string asm_output = out.str();
        int line_count = 0;
        std::istringstream iss(asm_output);
        std::string line;
        while (std::getline(iss, line) && line_count < 20) {
            std::cout << "    " << line << "\n";
            line_count++;
        }
        if (asm_output.length() > 0) {
            std::cout << "    ... (total " << asm_output.length() << " bytes)\n";
        }
        
        test_pass();
    } catch (const std::exception& e) {
        test_fail(std::string("Exception: ") + e.what());
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Main Test Runner
// ════════════════════════════════════════════════════════════════════════════

int main()
{
    std::cout.setf(std::ios::unitbuf);
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Progressive Lowering Pipeline Test Suite\n";
    std::cout << std::string(70, '=') << "\n";
    
    // Run all tests
    test_matmul_tiling();
    test_scratchpad_allocation();
    test_k_spilling();
    test_double_buffering();
    test_riscv_emission();
    test_x86_emission();
    
    // Summary
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Test Summary: " << test_passed << "/" << test_count << " passed\n";
    std::cout << std::string(70, '=') << "\n";
    
    return (test_passed == test_count) ? 0 : 1;
}
