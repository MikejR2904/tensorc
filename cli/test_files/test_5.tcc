import std;
import tensor as ts;

async fn residual_bottleneck(x: Tensor#(f32, [B, D]), w1: Tensor#(f32, [D, D]), w2: Tensor#(f32, [D, D])) -> Tensor#(f32, [B, D]) {
    let branch_a = spawn ts::matmul(x, w1);
    let branch_b = spawn ts::matmul(x, w2);
    
    let out = ts::relu(await branch_a + await branch_b);
    return out + x;
}

async fn main() {
    let grad params_w1: Tensor#(f32, [512, 512]) = ts::randn([512, 512]);
    let grad params_w2: Tensor#(f32, [512, 512]) = ts::randn([512, 512]);
    let input: Tensor#(f32, [64, 512]) = ts::ones([64, 512]);

    let task_outer = spawn residual_bottleneck(input, params_w1, params_w2);

    let other_work = spawn ts::sum(input);
    
    let final_output = await task_outer;
    let checksum = await other_work;

    let scaled_output = final_output * 0.5;

    if (scaled_output.requires_grad && checksum > 0.0) {
        std::println("Stress Test Phase 1: Grad & Concurrency OK");
    }

    let mean_val = ts::mean(scaled_output);
    if (mean_val > 100.0) {
        let grad adjustment = ts::ones([64, 512]);
        let _ = await spawn ts::matmul(scaled_output, adjustment);
        std::println("Branch: High activation path");
    } else {
        std::println("Branch: Nominal activation path");
    }
}