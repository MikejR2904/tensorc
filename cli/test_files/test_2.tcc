import std;
import tensor as ts;

struct ModelConfig {
    learning_rate: f32,
    batch_size: i32,
    metadata: str
}

async fn compute_layer(input: Tensor<f32, [N, 512]>, weights: Tensor<f32, [512, 256]>) -> Tensor<f32, [N, 256]> {
    // Testing Matrix Multiplication and result shape inference
    let output = input @ weights;
    return output;
}

async fn main() {
    // 1. Setup Configuration
    let config = ModelConfig { 
        learning_rate: 0.001, 
        batch_size: 32, 
        metadata: "TensorC-Alpha" 
    };

    // 2. Initialize Tensors with Symbolic Dimensions
    // N is symbolic, batch_size is concrete
    let data: Tensor<f32, [32, 512]> = ts::ones([32, 512]);
    let w1:   Tensor<f32, [512, 256]> = ts::rand([512, 256]);
    let w2:   Tensor<f32, [256, 128]> = ts::rand([256, 128]);

    std::println("Starting distributed inference for: " + config.metadata);

    // 3. Testing Nested Concurrency (Spawn/Await)
    // We spawn two layers to run in parallel on different 'workers'
    let task1 = spawn compute_layer(data, w1);
    
    // While layer 1 is crunching, we can do scalar math
    let alpha = 0.5 * config.learning_rate;
    
    // Halt for layer 1
    let layer1_out = await task1;

    // 4. Test Broadcasting & Binary Ops
    // Adding a scalar to a tensor should trigger your scalar-tensor logic
    let biased_layer = layer1_out + alpha;

    // 5. Final Layer
    let task2 = spawn compute_layer(biased_layer, w2);
    let final_result = await task2;

    // 6. Verification
    if (final_result.shape[1] == 128) {
        std::println("Inference complete. Output shape matched.");
    } else {
        std::println("Error: Shape mismatch in pipeline.");
    }
}