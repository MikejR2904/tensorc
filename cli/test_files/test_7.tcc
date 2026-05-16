import std;
import math;
import tensor as ts;

struct Config {
    threshold: f32,
    multiplier: f32
}

struct Result {
    id: i32,
    value: f32,
    status: i32
}

struct AnalysisData {
    sum: f32,
    mean: f32,
    max: f32,
    threshold: f32
}

// Helper function to process a single value with threshold check
fn process_value(val: f32, cfg: Config) -> f32 {
    if (val > cfg.threshold) {
        return val * cfg.multiplier;
    }
    return 0.0;
}

// Process multiple values in sequence
fn batch_process(v1: f32, v2: f32, v3: f32, cfg: Config) -> f32 {
    let r1 = process_value(v1, cfg);
    let r2 = process_value(v2, cfg);
    let r3 = process_value(v3, cfg);
    let sum = r1 + r2 + r3;
    return sum;
}

// Async processing with struct parameter
async fn async_process(data: Tensor<f32>, cfg: Config) -> f32 {
    let max_val = ts::max(data);
    let processed = process_value(max_val, cfg);
    return processed;
}

// Function returning a struct
fn create_result(id: i32, value: f32) -> Result {
    let res = Result { id: id, value: value, status: 1 };
    return res;
}

// Extract and process result struct
fn check_result(res: Result) -> f32 {
    if (res.status > 0) {
        return res.value * 2.0;
    }
    return 0.0;
}

// Complex struct with multiple fields
fn analyze_data(data: Tensor<f32>, cfg: Config) -> AnalysisData {
    let s = ts::sum(data);
    let m = ts::mean(data);
    let mx = ts::max(data);
    let analysis = AnalysisData { 
        sum: s, 
        mean: m, 
        max: mx, 
        threshold: cfg.threshold 
    };
    return analysis;
}

// Process analysis result with multiple field accesses
fn score_analysis(analysis: AnalysisData, multiplier: f32) -> f32 {
    let threshold = analysis.threshold;
    let max_val = analysis.max;
    
    if (max_val > threshold) {
        return max_val * multiplier;
    }
    return 0.0;
}

async fn main() -> void {
    // Test 1: Basic struct creation and field access
    let cfg = Config { threshold: 0.5, multiplier: 2.5 };
    std::println("Configuration created");

    // Test 2: Tensor operations
    let shape = [4, 4];
    let data = ts::rand(shape);
    std::println("Data generated");

    // Test 3: Single value processing with struct
    let single_result = process_value(0.8, cfg);
    
    // Test 4: Batch processing multiple values
    let batch_value = batch_process(0.3, 0.6, 0.9, cfg);
    
    // Test 5: Async operation with spawn/await
    let task = spawn async_process(data, cfg);
    let async_value = await task;
    
    // Test 6: Struct return and field access
    let res = create_result(42, async_value);
    let final_value = check_result(res);
    
    // Test 7: Complex struct processing with multiple fields
    let analysis = analyze_data(data, cfg);
    let score = score_analysis(analysis, cfg.multiplier);
    
    // Test 8: Conditional logic on struct fields
    let result_id = res.id;
    if (result_id > 0) {
        std::println("Processing complete");
    }
    
    // Test 9: Multiple struct instances with different values
    let cfg2 = Config { threshold: 0.3, multiplier: 1.5 };
    let result2 = process_value(0.5, cfg2);
    let batch2 = batch_process(0.2, 0.4, 0.7, cfg2);
}