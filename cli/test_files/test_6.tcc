import std;
import tensor as ts;

// A function that returns another Task (Nested Async)
async fn get_processor(mode: i32) -> Task#(Tensor#(f32, [512, 512])) {
    if (mode > 0) {
        return spawn ts::eye([512, 512]);
    }
    return spawn ts::zeros([512, 512]);
}

async fn main() {
    let dim = 512;
    let grad weights: Tensor#(f32, [dim, dim]) = ts::randn([dim, dim]);
    
    // 1. STRESS: Double Await 
    // This checks if parseUnary() recurses correctly: await (await get_processor(1))
    let identity_matrix = await await get_processor(1);

    // 2. STRESS: Inline Spawn & Await inside Binary Ops
    // The compiler must resolve the spawn, then the await, then the addition
    // all while keeping track of the 'grad' flag from 'weights'
    let complex_result = (await spawn ts::relu(weights)) + (await spawn ts::sigmoid(weights));

    // 3. STRESS: The "Pipeline" Operator with Async
    // If you implemented PIPE (|>) as we discussed earlier, let's chain them.
    // Each step involves a Tensor# metadata check.
    let pipeline_out = weights 
        |> ts::matmul(identity_matrix) 
        |> ts::relu() 
        |> ts::sum();

    // 4. STRESS: Distributed Worker Spawning with Captured Context
    let factor = 0.5;
    let worker_task = spawn async fn(x: Tensor) -> Tensor {
        // This tests if 'factor' is correctly captured in the distributed closure
        return x * factor;
    }(complex_result);

    let final_tensor = await worker_task;

    // 5. VALIDATION: Check for "Poisoned" Types
    // If the metadata [dim, dim] survived all these spawns/awaits, this passes.
    if (final_tensor.shape[0] == 512 && final_tensor.requires_grad) {
        std::println("Ultimate Test: ALL PASS");
    } else {
        std::println("Ultimate Test: Metadata or Grad-bit corrupted.");
    }
}