import std;
import tensor as ts;

// 1. Test generic async function with symbolic shapes
async fn process_block(data: Tensor<f32, [N, D]>, weight: Tensor<f32, [D, H]>) -> Tensor<f32, [N, H]> {
    // Test matrix multiplication and bias addition
    let grad bias = ts::zeros([H]);
    let output = (data @ weight) + bias;
    return ts::relu(output);
}

// 2. Test a function that returns a primitive inside an async context
async fn compute_metric(val: Tensor) -> f32 {
    return ts::sum(val) / 2.0;
}

async fn main() {
    // 3. Test 'grad' flag on complex types
    let grad weights: Tensor<f32, [512, 256]> = ts::randn([512, 256]);
    let input: Tensor<f32, [32, 512]> = ts::ones([32, 512]);

    std::println("Starting parallel execution...");

    // 4. Test Nested Concurrency (Spawn within Main)
    // We spawn two separate 'process_block' tasks
    let t1 = spawn process_block(input, weights);
    let t2 = spawn process_block(input, weights);

    // 5. Test Await and type unwrapping
    // result1 and result2 should be Tensor<f32, [32, 256]>
    let result1 = await t1;
    let result2 = await t2;

    // 6. Test Autograd Propagation
    // result1 came from a path involving 'weights' (grad), so it should have it too
    let combined = result1 + result2;
    
    // 7. Test Spawning a Task that depends on a previous Await
    let metric_task = spawn compute_metric(combined);
    let final_score = await metric_task;

    // 8. Test Logic & Comparison
    if (final_score > 0.0) {
        std::println("Model output valid.");
    } else {
        std::println("Warning: Dead neurons detected.");
    }

    // 9. Verify the grad flag survived the Task/Await cycle
    if (combined.requires_grad) {
        std::println("Autograd: Success (Flag propagated through Spawn/Await)");
    } else {
        std::println("Autograd: Failed (Flag lost in Task wrapper)");
    }
}