/// codegen/tools/ExecutionHarness.h
/// 
/// Harness for loading and executing compiled ELF binaries.
/// Provides mechanisms to:
/// - Load ELF binaries into memory
/// - Locate and call functions
/// - Pass arguments and capture return values
/// - Detect crashes and errors
/// - Validate outputs

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstring>
#include <cstdint>

namespace codegen::tools {

/// Result of executing a function
struct ExecutionResult {
    bool success = true;
    std::string error_message;
    
    // For scalar return values
    double scalar_result = 0.0;
    int64_t int_result = 0;
    
    // For array/tensor results
    std::vector<double> array_result;
    std::vector<int64_t> int_array_result;
    
    // Execution time in milliseconds
    double execution_time_ms = 0.0;
};

/// Harness for executing compiled code
class ExecutionHarness {
public:
    ExecutionHarness() = default;
    ~ExecutionHarness() = default;
    
    /// Load an ELF binary from file
    bool load_elf(const std::string& filepath);
    
    /// Load ELF binary from memory
    bool load_elf_memory(const void* data, size_t size);
    
    /// Get information about loaded binary
    std::string get_loaded_binary_info() const;
    
    /// Execute function with double arguments and double return
    ExecutionResult execute_f64_f64(
        const std::string& func_name,
        double arg1);
    
    /// Execute function with two double arguments
    ExecutionResult execute_f64_f64_f64(
        const std::string& func_name,
        double arg1, double arg2);
    
    /// Execute function with three double arguments
    ExecutionResult execute_f64_f64_f64_f64(
        const std::string& func_name,
        double arg1, double arg2, double arg3);
    
    /// Execute function with vector arguments
    ExecutionResult execute_vector_add(
        const std::string& func_name,
        const std::vector<double>& a,
        const std::vector<double>& b);
    
    /// Execute function with matrix arguments (MatMul)
    ExecutionResult execute_matmul(
        const std::string& func_name,
        const std::vector<double>& A,
        const std::vector<double>& B,
        size_t rows_A, size_t cols_A,
        size_t rows_B, size_t cols_B);
    
    /// Execute activation function (scalar)
    ExecutionResult execute_activation(
        const std::string& func_name,
        double input);
    
    /// Execute element-wise operation on arrays
    ExecutionResult execute_elemwise(
        const std::string& func_name,
        const std::vector<double>& inputs);
    
    /// Execute reduction (sum/mean/max/min)
    ExecutionResult execute_reduction(
        const std::string& func_name,
        const std::vector<double>& inputs);
    
    /// Set verbose logging
    void set_verbose(bool verbose) { verbose_ = verbose; }
    
    /// Get last execution error
    std::string get_last_error() const { return last_error_; }

private:
    std::string binary_path_;
    std::string last_error_;
    bool verbose_ = false;
    bool loaded_ = false;
    void* elf_image_ = nullptr;
    size_t elf_size_ = 0;
    
    /// Helper: Find function by name in loaded binary
    void* find_function(const std::string& name);
    
    /// Helper: Execute with signal handling for segfaults
    void execute_with_protection(
        std::function<void()> fn,
        ExecutionResult& result);
};

} // namespace codegen::tools
