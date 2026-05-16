// Debug test for math module
import math;

async fn main() {
    let x = 5.0;
    // Try calling sqrt with module prefix
    let result = math::sqrt(x);
    std::println("Done");
}
