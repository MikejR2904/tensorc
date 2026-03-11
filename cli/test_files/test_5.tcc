import std;
import tensor as ts;

// A complex block representing a Residual-like connection with a bottleneck
async fn residual_bottleneck(x: Tensor#(f32, [B, D]), w1: Tensor#(f32, [D, D]), w2: Tensor#(f32, [D, D])) -> Tensor#(f32, [B, D]) {
    // Branch A: Direct path
    let branch_a = spawn ts::matmul(x, w1);
    
    // Branch B: Processed path
    let branch_b = spawn ts::matmul(x, w2);
    
    // Sync branches and apply non-linearity
    let out = ts::relu(await branch_a + await branch_b);
    
    // Residual connection
    return out + x;
}

async fn main() {
    // 1. SETUP: High-dimensional tensors
    let grad params_w1: Tensor#(f32, [512, 512]) = ts::randn([512, 512]);
    let grad params_w2: Tensor#(f32, [512, 512]) = ts::randn([512, 512]);
    let input: Tensor#(f32, [64, 512]) = ts::ones([64, 512]);

    // 2. NESTED SPAWNS: Spawning a task that spawns its own internal tasks
    // This tests the stack depth and the async state machine's ability to yield
    let task_outer = spawn residual_bottleneck(input, params_w1, params_w2);

    // 3. PIPELINE PARALLELISM: While the bottleneck is running, do something else
    let other_work = spawn ts::sum(input);
    
    // 4. THE DIAMOND MERGE:
    // Awaiting the nested structure. The grad flag from params_w1 AND params_w2 
    // must converge into 'final_output'.
    let final_output = await task_outer;
    let checksum = await other_work;

    // 5. TRICKY SEMANTICS: In-place-like operations and Grad check
    // Even if we scale the output, the grad graph must remain intact.
    let scaled_output = final_output * 0.5;

    // 6. VALIDATION
    if (scaled_output.requires_grad && checksum > 0.0) {
        std::println("Stress Test Phase 1: Grad & Concurrency OK");
    }

    // 7. DATA-DEPENDENT CONTROL FLOW:
    // This forces the compiler to handle potentially dynamic graph paths
    let mean_val = ts::mean(scaled_output);
    if (mean_val > 100.0) {
        let grad adjustment = ts::ones([64, 512]);
        let _ = await spawn ts::matmul(scaled_output, adjustment);
        std::println("Branch: High activation path");
    } else {
        std::println("Branch: Nominal activation path");
    }
}