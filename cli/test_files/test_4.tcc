import std;
import tensor as ts;

// Function with symbolic shape and autograd using Hash sigil
async fn train_step(x: Tensor#(f32, [B, D]), w: Tensor#(f32, [D, H])) -> Tensor#(f32, [B, H]) {
    // b is initialized with 'grad' context
    let grad b = ts::zeros([H]);
    let out = (x @ w) + b;
    return ts::relu(out);
}

async fn main() {
    // Defining dimensions to satisfy Semantic Analyzer
    let B = 32;
    let D = 128;
    let H = 64;

    // Testing 'grad' on a full declaration with Hash sigil
    let grad weights: Tensor#(f32, [D, H]) = ts::randn([D, H]);
    let input: Tensor#(f32, [B, D]) = ts::ones([B, D]);

    // Testing Spawn - logic for async task distribution
    let task = spawn train_step(input, weights);
    
    // Testing Await - unwrapping Task<Tensor> to Tensor
    let result = await task;

    // Verify 'requires_grad' survived the Spawn/Await round-trip
    if (result.requires_grad) {
        std::println("Autograd flag propagated successfully!");
    } else {
        std::println("Error: Autograd flag lost during task execution.");
    }
}