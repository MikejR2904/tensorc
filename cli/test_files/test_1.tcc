// 1. Testing Imports and Aliasing
import std;
import math;
import tensor as ts;

// 2. Testing Structs and Dot Access
struct Particle {
    x: f32,
    y: f32,
    mass: f32
}

async fn compute_momentum(p: Particle, velocity: f32) -> f32 {
    // Testing math module and dot access
    let speed = math::sqrt(velocity * velocity);
    return p.mass * speed;
}

async fn main() {
    // 3. Testing Tensor Operations
    // Using the 'ts' alias from 'import tensor as ts'
    let shape = [2, 2];
    let weights = ts::rand(shape);
    let inputs  = ts::ones(shape);
    
    // 4. Testing Distributed Concurrency (Spawn/Await)
    let p1 = Particle { x: 0.0, y: 1.2, mass: 10.5 };
    
    // Spawn a task on a background worker
    let task = spawn compute_momentum(p1, 5.5);
    
    std::print("Calculating momentum...");
    
    // Halt until result is ready
    let result = await task;
    
    // 5. Testing Logic and Built-ins
    if (result > 50.0) {
        std::println("High momentum detected");
    } else {
        std::println("Normal momentum");
    }
}