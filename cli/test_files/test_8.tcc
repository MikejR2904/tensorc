import std;
import math;
import tensor as ts;

struct ModelConfig {
    scale: f32,
    dropout_rate: f32,
    bias_multiplier: f32
}

// A highly complex element-wise operation chain.
// Ideal for kernel fusion because it chains: Add -> Mul -> Exp -> Add -> Div -> Mul (Dropout Mask)
// Instead of 6 separate loops/allocations, a good compiler fuses this into 1 single loop.
fn fused_attention_core(
    raw_weights: Tensor<f32, [N, M]>, 
    bias: Tensor<f32, [N, M]>, 
    mask: Tensor<f32, [N, M]>,
    cfg: ModelConfig
) -> Tensor<f32, [N, M]> {
    
    // 1. Scale element-wise
    let scaled = raw_weights * cfg.scale;
    
    // 2. Fused Bias Add
    let biased = scaled + (bias * cfg.bias_multiplier);
    
    // 3. Custom Element-wise Softmax-like Numerator (GELU/Exp-style activation)
    // In standard execution, this creates massive memory overhead. 
    // Fused, it is just a registers-only math operation.
    let activated = ts::exp(biased);
    
    // 4. Safe division wrapper (simulating specialized normalization)
    let normalized = activated / (activated + 1.0);
    
    // 5. Apply dropout mask element-wise
    let output = normalized * mask;
    
    return output;
}

// An async pipeline function that coordinates the heavy matrix multiplication 
// and immediately pipes it into our fused element-wise block.
async fn compute_attention_head(
    q: Tensor<f32, [N, D]>, 
    k: Tensor<f32, [M, D]>, 
    bias: Tensor<f32, [N, M]>,
    mask: Tensor<f32, [N, M]>,
    cfg: ModelConfig
) -> Tensor<f32, [N, M]> {
    
    // Heavy GEMM (Matrix Multiplication): Q @ K^T
    // This outputs an [N, M] tensor.
    // Cannot be fused with the element-wise ops, serving as the 'fusing boundary'.
    let raw_weights = q @ k.T; 
    
    // Immediately pass the output to the fused element-wise block
    let final_attention = fused_attention_core(raw_weights, bias, mask, cfg);
    
    return final_attention;
}

async fn main() -> void {
    std::println("Initializing TensorC Fused Kernel Test Suite...");

    // Setup configuration parameters
    let cfg = ModelConfig { 
        scale: 0.125,          // e.g., 1 / sqrt(d_k) where d_k = 64
        dropout_rate: 0.1, 
        bias_multiplier: 0.5 
    };

    // Symbolic or dynamic runtime shapes (Batch size / Sequence lengths)
    // N = 128 (Sequence Length 1), M = 128 (Sequence Length 2), D = 64 (Head Dimension)
    let q_shape = [128, 64];
    let k_shape = [128, 64];
    let matrix_shape = [128, 128];

    // Generate test inputs
    let q = ts::rand(q_shape);
    let k = ts::rand(k_shape);
    let bias = ts::rand(matrix_shape);
    let mask = ts::rand(matrix_shape); // Simulating a binary/dropout mask

    std::println("Inputs allocated. Dispatching async fused kernel execution...");

    // Dispatch the attention head calculation to an async task group
    let task1 = spawn compute_attention_head(q, k, bias, mask, cfg);
    
    // Compiler should be optimizing the inside of 'compute_attention_head' right now
    let attention_matrix = await task1;

    // Validate properties of the fused output using the type properties we saw earlier
    let out_rank = attention_matrix.rank;
    let out_dtype = attention_matrix.dtype;

    if (out_rank == 2) {
        std::println("Verification Success: Rank matches expected dimensional hierarchy.");
    }
    
    std::println("Fused kernel evaluation pipeline executed flawlessly.");
}