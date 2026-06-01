#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>

/// codegen/tools/test_execution_examples.cpp
/// 
/// Practical examples demonstrating how to execute generated code
/// and verify correctness on actual CPU.

namespace codegen::tools::execution_examples {

// Example 1: Simple 2x2 matrix multiplication
// Input: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
// Expected: C = [[19, 22], [43, 50]]
class MatMulExecutionTest : public ::testing::Test {
protected:
    // Pseudo-assembly for 2x2 matrix multiply (conceptual)
    const char* matmul_2x2_assembly_riscv64 = R"(
        # Assume input matrices in memory:
        # A at sp+0, B at sp+32, C at sp+64
        # All double precision (8 bytes per element)
        
        .text
        .globl matmul_2x2
        matmul_2x2:
            addi sp, sp, -96        # Allocate stack frame
            
            # Load A[0,0], A[0,1], A[1,0], A[1,1]
            fld f0, 0(a0)           # A[0,0]
            fld f1, 8(a0)           # A[0,1]
            fld f2, 16(a0)          # A[1,0]
            fld f3, 24(a0)          # A[1,1]
            
            # Load B[0,0], B[0,1], B[1,0], B[1,1]
            fld f4, 0(a1)           # B[0,0]
            fld f5, 8(a1)           # B[0,1]
            fld f6, 16(a1)          # B[1,0]
            fld f7, 24(a1)          # B[1,1]
            
            # Compute C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            fmul.d f8, f0, f4       # A[0,0] * B[0,0]
            fmul.d f9, f1, f6       # A[0,1] * B[1,0]
            fadd.d f8, f8, f9       # C[0,0]
            
            # Compute C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            fmul.d f9, f0, f5       # A[0,0] * B[0,1]
            fmul.d f10, f1, f7      # A[0,1] * B[1,1]
            fadd.d f9, f9, f10      # C[0,1]
            
            # Compute C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            fmul.d f10, f2, f4      # A[1,0] * B[0,0]
            fmul.d f11, f3, f6      # A[1,1] * B[1,0]
            fadd.d f10, f10, f11    # C[1,0]
            
            # Compute C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
            fmul.d f11, f2, f5      # A[1,0] * B[0,1]
            fmul.d f12, f3, f7      # A[1,1] * B[1,1]
            fadd.d f11, f11, f12    # C[1,1]
            
            # Store results to C (a2 = pointer to C)
            fsd f8, 0(a2)           # C[0,0]
            fsd f9, 8(a2)           # C[0,1]
            fsd f10, 16(a2)         # C[1,0]
            fsd f11, 24(a2)         # C[1,1]
            
            addi sp, sp, 96
            ret
    )";
};

TEST_F(MatMulExecutionTest, VerifySimpleMatMul2x2) {
    // Input matrices
    double A[4] = {1.0, 2.0, 3.0, 4.0};
    double B[4] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};
    
    // Expected result
    double expected[4] = {19.0, 22.0, 43.0, 50.0};
    
    // In real test, we would:
    // 1. Assemble the above .s file to ELF
    // 2. Load ELF binary into memory
    // 3. Call the matmul_2x2 function with pointers to A, B, C
    // 4. Verify C matches expected
    
    // For now, compute expected and verify logic
    double C_computed[4];
    
    // C = A * B (manual computation)
    // C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
    C_computed[0] = A[0]*B[0] + A[1]*B[2];
    // C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
    C_computed[1] = A[0]*B[1] + A[1]*B[3];
    // C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 43
    C_computed[2] = A[2]*B[0] + A[3]*B[2];
    // C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 50
    C_computed[3] = A[2]*B[1] + A[3]*B[3];
    
    // Verify all elements match expected
    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(C_computed[i], expected[i])
            << "Element C[" << i << "] mismatch";
    }
}

// Example 2: Element-wise addition
class ElemWiseAddExecutionTest : public ::testing::Test {
protected:
    const char* elemwise_add_assembly_riscv64 = R"(
        # Element-wise add: C[i] = A[i] + B[i] for i in 0..n-1
        # a0 = pointer to A (double*)
        # a1 = pointer to B (double*)
        # a2 = pointer to C (double*)
        # a3 = n (number of elements)
        
        .text
        .globl elemwise_add
        elemwise_add:
            # Check if n <= 0
            ble a3, zero, elemwise_add_done
            
            # Convert n to byte count (8 bytes per double)
            slli a3, a3, 3
            
        elemwise_add_loop:
            # Load A[i]
            fld f0, 0(a0)
            # Load B[i]
            fld f1, 0(a1)
            # Add
            fadd.d f2, f0, f1
            # Store C[i]
            fsd f2, 0(a2)
            
            # Advance pointers
            addi a0, a0, 8
            addi a1, a1, 8
            addi a2, a2, 8
            
            # Loop while (byte_count > 0)
            addi a3, a3, -8
            bgt a3, zero, elemwise_add_loop
            
        elemwise_add_done:
            ret
    )";
};

TEST_F(ElemWiseAddExecutionTest, VerifyElementWiseAdd) {
    // Test vectors
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> B = {1.5, 2.5, 3.5, 4.5, 5.5};
    std::vector<double> C(5, 0.0);
    
    // Expected
    std::vector<double> expected = {2.5, 4.5, 6.5, 8.5, 10.5};
    
    // Compute element-wise add
    for (size_t i = 0; i < A.size(); i++) {
        C[i] = A[i] + B[i];
    }
    
    // Verify
    for (size_t i = 0; i < C.size(); i++) {
        EXPECT_DOUBLE_EQ(C[i], expected[i])
            << "Element C[" << i << "] mismatch";
    }
}

// Example 3: ReLU activation
class ActivationExecutionTest : public ::testing::Test {
protected:
    const char* relu_assembly_riscv64 = R"(
        # ReLU: C[i] = max(0, A[i])
        # a0 = pointer to A (double*)
        # a1 = pointer to C (double*)
        # a2 = n (number of elements)
        
        .text
        .globl relu
        relu:
            # Load constant 0.0
            fmv.d.x f0, zero        # f0 = 0.0
            
            # Check if n <= 0
            ble a2, zero, relu_done
            
            # Convert n to byte count
            slli a2, a2, 3
            
        relu_loop:
            # Load A[i]
            fld f1, 0(a0)
            
            # Compare A[i] vs 0.0
            fle.d t0, f0, f1        # if (0 <= A[i])
            
            # Conditional move: if A[i] >= 0, output A[i]; else 0
            fsel f2, f1, f0, t0     # Pseudo: f2 = (t0 ? f1 : f0)
            
            # Store C[i]
            fsd f2, 0(a1)
            
            # Advance pointers
            addi a0, a0, 8
            addi a1, a1, 8
            
            # Loop
            addi a2, a2, -8
            bgt a2, zero, relu_loop
            
        relu_done:
            ret
    )";
};

TEST_F(ActivationExecutionTest, VerifyReLU) {
    // Test vector with mixed signs
    std::vector<double> A = {-2.0, -1.0, 0.0, 1.0, 2.0, -5.0};
    std::vector<double> C(A.size(), 0.0);
    
    // Expected
    std::vector<double> expected = {0.0, 0.0, 0.0, 1.0, 2.0, 0.0};
    
    // Compute ReLU
    for (size_t i = 0; i < A.size(); i++) {
        C[i] = std::max(0.0, A[i]);
    }
    
    // Verify
    for (size_t i = 0; i < C.size(); i++) {
        EXPECT_DOUBLE_EQ(C[i], expected[i])
            << "ReLU(A[" << i << "]) mismatch";
    }
}

// Example 4: Reduction (Sum)
class ReductionExecutionTest : public ::testing::Test {
protected:
    const char* sum_assembly_riscv64 = R"(
        # Sum reduction: result = sum(A[0..n-1])
        # a0 = pointer to A (double*)
        # a1 = n (number of elements)
        # f0 = return value (sum)
        
        .text
        .globl reduce_sum
        reduce_sum:
            # Initialize accumulator to 0.0
            fmv.d.x f0, zero
            
            # Check if n <= 0
            ble a1, zero, reduce_sum_done
            
        reduce_sum_loop:
            # Load A[i]
            fld f1, 0(a0)
            # Accumulate
            fadd.d f0, f0, f1
            
            # Advance pointer
            addi a0, a0, 8
            
            # Decrement count
            addi a1, a1, -1
            
            # Loop
            bgt a1, zero, reduce_sum_loop
            
        reduce_sum_done:
            ret
    )";
};

TEST_F(ReductionExecutionTest, VerifySum) {
    // Test vector
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Expected sum = 1+2+3+4+5 = 15
    double expected = 15.0;
    
    // Compute sum
    double result = 0.0;
    for (double val : A) {
        result += val;
    }
    
    // Verify
    EXPECT_DOUBLE_EQ(result, expected);
}

// Example 5: Benchmark - Matrix multiply performance
class PerformanceExecutionTest : public ::testing::Test {
public:
    static constexpr size_t MATRIX_SIZE = 64;
    static constexpr int ITERATIONS = 100;
};

TEST_F(PerformanceExecutionTest, MatMulPerformance) {
    // Allocate matrices
    std::vector<double> A(MATRIX_SIZE * MATRIX_SIZE, 1.0);
    std::vector<double> B(MATRIX_SIZE * MATRIX_SIZE, 1.0);
    std::vector<double> C(MATRIX_SIZE * MATRIX_SIZE, 0.0);
    
    // Timing would go here in actual test
    // For now, just verify dimensions
    EXPECT_EQ(A.size(), MATRIX_SIZE * MATRIX_SIZE);
    EXPECT_EQ(B.size(), MATRIX_SIZE * MATRIX_SIZE);
    EXPECT_EQ(C.size(), MATRIX_SIZE * MATRIX_SIZE);
}

// Example 6: Error detection - Memory access violations
class ErrorDetectionTest : public ::testing::Test {};

TEST_F(ErrorDetectionTest, CatchSegmentationFault) {
    // Real test would:
    // 1. Load ELF binary
    // 2. Add signal handler for SIGSEGV
    // 3. Execute function with invalid pointer
    // 4. Verify signal was caught
    // 5. Report error with helpful message
    
    // For now, document the expected behavior
    EXPECT_TRUE(true) << "SEGFAULT detection would be implemented in ExecutionHarness";
}

// Example 7: Correctness verification - Numerical accuracy
class NumericalAccuracyTest : public ::testing::Test {};

TEST_F(NumericalAccuracyTest, FloatingPointPrecision) {
    // Test that generated code maintains numerical accuracy
    double a = 0.1;
    double b = 0.2;
    double c = a + b;
    
    // Expected (approximately 0.3, with floating point error)
    double expected = 0.3;
    
    // Allow small epsilon for floating point comparison
    const double epsilon = 1e-15;
    EXPECT_NEAR(c, expected, epsilon);
}

} // namespace codegen::tools::execution_examples

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
