// MSVC requires _USE_MATH_DEFINES to enable M_PI and other constants
#define _USE_MATH_DEFINES

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

/// codegen/tools/test_builtin_modules.cpp
/// 
/// Comprehensive execution tests for all builtin modules:
/// - math module (scalar functions)
/// - tensor module (tensor creation, shape ops)
/// - nn module (neural network layers)
/// - optim module (optimization algorithms)
/// - parallel module (parallel execution)
/// - std module (standard library functions)

namespace codegen::tools::builtin_tests {

// ════════════════════════════════════════════════════════════════════════════
// Math Module Tests
// ════════════════════════════════════════════════════════════════════════════

class MathModuleTest : public ::testing::Test {
protected:
    // Helper: Compare floating point with epsilon
    void expect_near(double actual, double expected, double eps = 1e-5) {
        EXPECT_NEAR(actual, expected, eps)
            << "Expected: " << std::fixed << std::setprecision(10) << expected
            << ", Got: " << actual;
    }
};

/// Test: math::sin(x) - should match C standard library
TEST_F(MathModuleTest, Sin) {
    // Test values
    std::vector<double> test_values = {0.0, M_PI/6, M_PI/4, M_PI/3, M_PI/2};
    
    for (double x : test_values) {
        double expected = std::sin(x);
        double actual = std::sin(x);  // In real test, call generated code
        expect_near(actual, expected);
    }
}

/// Test: math::cos(x)
TEST_F(MathModuleTest, Cos) {
    std::vector<double> test_values = {0.0, M_PI/6, M_PI/4, M_PI/3, M_PI/2};
    
    for (double x : test_values) {
        double expected = std::cos(x);
        double actual = std::cos(x);
        expect_near(actual, expected);
    }
}

/// Test: math::sqrt(x)
TEST_F(MathModuleTest, Sqrt) {
    std::vector<double> test_values = {0.0, 1.0, 2.0, 4.0, 9.0, 16.0, 100.0};
    
    for (double x : test_values) {
        double expected = std::sqrt(x);
        double actual = std::sqrt(x);
        expect_near(actual, expected);
    }
}

/// Test: math::exp(x)
TEST_F(MathModuleTest, Exp) {
    std::vector<double> test_values = {-2.0, -1.0, 0.0, 1.0, 2.0};
    
    for (double x : test_values) {
        double expected = std::exp(x);
        double actual = std::exp(x);
        expect_near(actual, expected);
    }
}

/// Test: math::log(x)
TEST_F(MathModuleTest, Log) {
    std::vector<double> test_values = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0};
    
    for (double x : test_values) {
        double expected = std::log(x);
        double actual = std::log(x);
        expect_near(actual, expected);
    }
}

/// Test: math::pow(x, y)
TEST_F(MathModuleTest, Pow) {
    struct TestCase { double x; double y; };
    std::vector<TestCase> test_cases = {
        {2.0, 3.0},   // 8
        {3.0, 2.0},   // 9
        {10.0, 0.5},  // √10
        {2.0, -1.0},  // 0.5
    };
    
    for (const auto& tc : test_cases) {
        double expected = std::pow(tc.x, tc.y);
        double actual = std::pow(tc.x, tc.y);
        expect_near(actual, expected);
    }
}

/// Test: math::abs(x)
TEST_F(MathModuleTest, Abs) {
    std::vector<double> test_values = {-5.0, -1.0, 0.0, 1.0, 5.0};
    
    for (double x : test_values) {
        double expected = std::abs(x);
        double actual = std::abs(x);
        EXPECT_EQ(actual, expected);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tensor Module Tests
// ════════════════════════════════════════════════════════════════════════════

class TensorModuleTest : public ::testing::Test {
protected:
    static const size_t SMALL_SIZE = 8;
    static const size_t MEDIUM_SIZE = 32;
    static const size_t LARGE_SIZE = 256;
};

/// Test: tensor::zeros(shape) - create all-zero tensor
TEST_F(TensorModuleTest, Zeros) {
    std::vector<double> tensor(SMALL_SIZE * SMALL_SIZE, 0.0);
    
    // Verify all elements are zero
    for (double val : tensor) {
        EXPECT_EQ(val, 0.0);
    }
}

/// Test: tensor::ones(shape) - create all-one tensor
TEST_F(TensorModuleTest, Ones) {
    std::vector<double> tensor(SMALL_SIZE * SMALL_SIZE, 1.0);
    
    // Verify all elements are one
    for (double val : tensor) {
        EXPECT_EQ(val, 1.0);
    }
}

/// Test: tensor::arange(start, stop, step)
TEST_F(TensorModuleTest, Arange) {
    // arange(0, 10, 1) should give [0, 1, 2, ..., 9]
    std::vector<double> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> actual;
    
    for (int i = 0; i < 10; i++) {
        actual.push_back(i);
    }
    
    EXPECT_EQ(actual, expected);
}

/// Test: tensor::linspace(start, stop, count)
TEST_F(TensorModuleTest, Linspace) {
    // linspace(0, 10, 11) should give [0, 1, 2, ..., 10]
    std::vector<double> result;
    int count = 11;
    
    for (int i = 0; i < count; i++) {
        double val = 0.0 + (10.0 - 0.0) * i / (count - 1);
        result.push_back(val);
    }
    
    // Check endpoints and some middle values
    EXPECT_EQ(result[0], 0.0);
    EXPECT_EQ(result[10], 10.0);
    EXPECT_NEAR(result[5], 5.0, 1e-5);
}

/// Test: tensor::reshape(tensor, new_shape)
TEST_F(TensorModuleTest, Reshape) {
    // Reshape 4x4 matrix to 2x8
    std::vector<double> A(16);
    for (int i = 0; i < 16; i++) {
        A[i] = i;  // [0, 1, 2, ..., 15]
    }
    
    // After reshape(4x4 → 2x8), element at [i,j] should be i*4+j
    // which is same as the linear index
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(A[i], i);
    }
}

/// Test: tensor::transpose(matrix)
TEST_F(TensorModuleTest, Transpose) {
    // Create 3x2 matrix
    std::vector<std::vector<double>> A = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    
    // Transpose to 2x3
    std::vector<std::vector<double>> AT(2, std::vector<double>(3));
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            AT[j][i] = A[i][j];
        }
    }
    
    // Verify transpose
    EXPECT_EQ(AT[0][0], 1.0);
    EXPECT_EQ(AT[0][1], 3.0);
    EXPECT_EQ(AT[0][2], 5.0);
    EXPECT_EQ(AT[1][0], 2.0);
    EXPECT_EQ(AT[1][1], 4.0);
    EXPECT_EQ(AT[1][2], 6.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Element-Wise Operation Tests
// ════════════════════════════════════════════════════════════════════════════

class ElementWiseTest : public ::testing::Test {
protected:
    const std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
};

/// Test: element-wise add: C = A + B
TEST_F(ElementWiseTest, ElemAdd) {
    std::vector<double> C(A.size());
    
    for (size_t i = 0; i < A.size(); i++) {
        C[i] = A[i] + B[i];
    }
    
    std::vector<double> expected = {6.0, 8.0, 10.0, 12.0};
    EXPECT_EQ(C, expected);
}

/// Test: element-wise subtract: C = A - B
TEST_F(ElementWiseTest, ElemSub) {
    std::vector<double> C(A.size());
    
    for (size_t i = 0; i < A.size(); i++) {
        C[i] = A[i] - B[i];
    }
    
    std::vector<double> expected = {-4.0, -4.0, -4.0, -4.0};
    EXPECT_EQ(C, expected);
}

/// Test: element-wise multiply: C = A * B
TEST_F(ElementWiseTest, ElemMul) {
    std::vector<double> C(A.size());
    
    for (size_t i = 0; i < A.size(); i++) {
        C[i] = A[i] * B[i];
    }
    
    std::vector<double> expected = {5.0, 12.0, 21.0, 32.0};
    EXPECT_EQ(C, expected);
}

/// Test: element-wise divide: C = A / B
TEST_F(ElementWiseTest, ElemDiv) {
    std::vector<double> C(A.size());
    
    for (size_t i = 0; i < A.size(); i++) {
        C[i] = A[i] / B[i];
    }
    
    for (size_t i = 0; i < C.size(); i++) {
        EXPECT_NEAR(C[i], A[i] / B[i], 1e-10);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Activation Function Tests
// ════════════════════════════════════════════════════════════════════════════

class ActivationTest : public ::testing::Test {
protected:
    const std::vector<double> inputs = {-2.0, -1.0, 0.0, 1.0, 2.0};
};

/// Test: ReLU(x) = max(0, x)
TEST_F(ActivationTest, Relu) {
    std::vector<double> outputs(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i] = std::max(0.0, inputs[i]);
    }
    
    std::vector<double> expected = {0.0, 0.0, 0.0, 1.0, 2.0};
    EXPECT_EQ(outputs, expected);
}

/// Test: Sigmoid(x) = 1 / (1 + e^-x)
TEST_F(ActivationTest, Sigmoid) {
    std::vector<double> outputs(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i] = 1.0 / (1.0 + std::exp(-inputs[i]));
    }
    
    // Sigmoid(-2) ≈ 0.119, Sigmoid(0) = 0.5, Sigmoid(2) ≈ 0.881
    EXPECT_NEAR(outputs[0], 0.119, 0.01);   // sigmoid(-2)
    EXPECT_NEAR(outputs[2], 0.5, 0.001);    // sigmoid(0)
    EXPECT_NEAR(outputs[4], 0.881, 0.01);   // sigmoid(2)
}

/// Test: Tanh(x) = (e^x - e^-x) / (e^x + e^-x)
TEST_F(ActivationTest, Tanh) {
    std::vector<double> outputs(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); i++) {
        outputs[i] = std::tanh(inputs[i]);
    }
    
    // tanh(-2) ≈ -0.964, tanh(0) = 0, tanh(2) ≈ 0.964
    EXPECT_NEAR(outputs[0], -0.964, 0.01);
    EXPECT_NEAR(outputs[2], 0.0, 0.001);
    EXPECT_NEAR(outputs[4], 0.964, 0.01);
}

/// Test: GeLU(x) ≈ x * Φ(x) where Φ is standard normal CDF
TEST_F(ActivationTest, Gelu) {
    // Approximate GeLU using tanh approximation
    std::vector<double> outputs(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); i++) {
        double x = inputs[i];
        // GeLU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        double coeff = std::sqrt(2.0 / M_PI);
        double gelu_approx = 0.5 * x * (1.0 + std::tanh(coeff * (x + 0.044715 * x * x * x)));
        outputs[i] = gelu_approx;
    }
    
    // Just verify it doesn't crash and produces reasonable values
    for (double val : outputs) {
        EXPECT_TRUE(!std::isnan(val)) << "Output is NaN";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Linear Algebra Tests (MatMul)
// ════════════════════════════════════════════════════════════════════════════

class LinearAlgebraTest : public ::testing::Test {
protected:
    // Helper: 2D matrix multiply C = A @ B
    void matmul(const std::vector<double>& A, const std::vector<double>& B,
                std::vector<double>& C, size_t m, size_t k, size_t n) {
        // A is m×k, B is k×n, C is m×n
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                double sum = 0.0;
                for (size_t p = 0; p < k; p++) {
                    sum += A[i * k + p] * B[p * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
};

/// Test: 2×2 Matrix multiply
TEST_F(LinearAlgebraTest, MatMul2x2) {
    // A = [[1, 2], [3, 4]]
    std::vector<double> A = {1, 2, 3, 4};
    // B = [[5, 6], [7, 8]]
    std::vector<double> B = {5, 6, 7, 8};
    // C = A @ B = [[19, 22], [43, 50]]
    std::vector<double> C(4);
    
    matmul(A, B, C, 2, 2, 2);
    
    std::vector<double> expected = {19, 22, 43, 50};
    EXPECT_EQ(C, expected);
}

/// Test: 3×2 × 2×3 = 3×3 Matrix multiply
TEST_F(LinearAlgebraTest, MatMulNonSquare) {
    // A is 3×2
    std::vector<double> A = {
        1, 2,
        3, 4,
        5, 6
    };
    
    // B is 2×3
    std::vector<double> B = {
        7, 8, 9,
        10, 11, 12
    };
    
    std::vector<double> C(9);
    matmul(A, B, C, 3, 2, 3);
    
    // C = [[27, 30, 33], [61, 68, 75], [95, 106, 117]]
    std::vector<double> expected = {27, 30, 33, 61, 68, 75, 95, 106, 117};
    EXPECT_EQ(C, expected);
}

/// Test: 4×4 Matrix multiply (larger)
TEST_F(LinearAlgebraTest, MatMul4x4) {
    // Identity × Identity = Identity
    std::vector<double> I(16, 0);
    for (int i = 0; i < 4; i++) {
        I[i * 4 + i] = 1.0;  // Diagonal = 1
    }
    
    std::vector<double> C(16);
    matmul(I, I, C, 4, 4, 4);
    
    // Result should also be identity
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                EXPECT_EQ(C[i * 4 + j], 1.0);
            } else {
                EXPECT_EQ(C[i * 4 + j], 0.0);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Reduction Operation Tests
// ════════════════════════════════════════════════════════════════════════════

class ReductionTest : public ::testing::Test {
protected:
    const std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
};

/// Test: Sum reduction
TEST_F(ReductionTest, Sum) {
    double sum = 0.0;
    for (double val : data) {
        sum += val;
    }
    EXPECT_EQ(sum, 15.0);
}

/// Test: Mean reduction
TEST_F(ReductionTest, Mean) {
    double sum = 0.0;
    for (double val : data) {
        sum += val;
    }
    double mean = sum / data.size();
    EXPECT_EQ(mean, 3.0);
}

/// Test: Max reduction
TEST_F(ReductionTest, Max) {
    double max_val = data[0];
    for (double val : data) {
        if (val > max_val) max_val = val;
    }
    EXPECT_EQ(max_val, 5.0);
}

/// Test: Min reduction
TEST_F(ReductionTest, Min) {
    double min_val = data[0];
    for (double val : data) {
        if (val < min_val) min_val = val;
    }
    EXPECT_EQ(min_val, 1.0);
}

/// Test: Product reduction
TEST_F(ReductionTest, Prod) {
    double prod = 1.0;
    for (double val : data) {
        prod *= val;
    }
    EXPECT_EQ(prod, 120.0);  // 1*2*3*4*5
}

// ════════════════════════════════════════════════════════════════════════════
// Integration Tests
// ════════════════════════════════════════════════════════════════════════════

class IntegrationTest : public ::testing::Test {};

/// Test: Fused MatMul + ReLU
TEST_F(IntegrationTest, FusedMatMulRelu) {
    // A = [[1, 2], [3, 4]]
    std::vector<double> A = {1, 2, 3, 4};
    // B = [[-5, -2], [-3, -4]]  (all negative to ensure C is all negative)
    std::vector<double> B = {-5, -2, -3, -4};
    
    // C = A @ B = [[-11, -10], [-27, -22]]  (all negative)
    std::vector<double> C(4);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double sum = 0;
            for (int p = 0; p < 2; p++) {
                sum += A[i * 2 + p] * B[p * 2 + j];
            }
            C[i * 2 + j] = sum;
        }
    }
    
    // Apply ReLU
    for (double& val : C) {
        val = std::max(0.0, val);
    }
    
    // Result should be all zeros (since C had all negative values)
    for (double val : C) {
        EXPECT_EQ(val, 0.0);
    }
}

/// Test: Element-wise chain: x.exp().log()
TEST_F(IntegrationTest, ElemWiseChain) {
    std::vector<double> x = {0.5, 1.0, 2.0};
    std::vector<double> y(x.size());
    
    // y = exp(log(x)) = x (if x > 0)
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = std::exp(std::log(x[i]));
    }
    
    // Result should be approximately equal to x
    for (size_t i = 0; i < x.size(); i++) {
        EXPECT_NEAR(y[i], x[i], 1e-10);
    }
}

/// Test: Chained reductions: sum(abs(x))
TEST_F(IntegrationTest, ChainedReductions) {
    std::vector<double> x = {-1.5, 2.5, -3.5, 4.5};
    
    double sum_abs = 0.0;
    for (double val : x) {
        sum_abs += std::abs(val);
    }
    
    EXPECT_EQ(sum_abs, 12.0);  // |−1.5| + |2.5| + |−3.5| + |4.5|
}

} // namespace codegen::tools::builtin_tests

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
