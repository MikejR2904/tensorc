import tensor as ts;

async fn fused_example(x: Tensor<f32>, w: Tensor<f32>) -> Tensor<f32> {
    let intermediate = x @ w;
    return ts::relu(intermediate);
}