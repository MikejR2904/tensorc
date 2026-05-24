# Building and Extending TensorC Modules
  
**Updated:** May 2026

---

## Table of Contents

1. [Module Types](#module-types)
2. [Built-in Modules Reference](#built-in-modules-reference)
3. [Creating Custom Modules](#creating-custom-modules)
4. [Module Best Practices](#module-best-practices)
5. [Extending Built-in Modules](#extending-built-in-modules)
6. [Publishing Modules](#publishing-modules)

---

## Module Types

### 1. Built-in Modules (Core)

Included with TensorC compiler. Optimized with custom IR lowering.

- **std**: I/O, assertions, basic utilities
- **math**: Mathematical functions and constants
- **tensor**: Tensor creation and operations
- **nn**: Neural network layers
- **optim**: Optimization algorithms
- **data**: Dataset utilities
- **parallel**: Concurrency and async operations

**Access:**
```tensorc
import std;          // Direct usage: std.println()
import math;         // math.sqrt(), math.pi
import tensor;       // tensor.zeros(), tensor.dot()
import tensor as ts; // Alias: ts.zeros(), ts.dot()
```

### 2. Standard Library Modules (Future)

Pre-built, high-quality modules distributed with TensorC.

- **linalg**: Linear algebra (eigenvalues, SVD, etc.)
- **stats**: Statistical functions
- **fft**: Fourier transforms
- **signal**: Signal processing
- **io**: Advanced file I/O

### 3. User-Defined Modules (Custom)

Written by developers for their projects.

```tensorc
import "./my_utils" as utils;
import "../shared/helpers" as helpers;
```

### 4. Third-Party Modules (Future)

Community modules published and installed via package manager.

```tensorc
import "ml.vision" as vision;        // External package
import "research.diffusion" as diff;  // Custom package
```

### Practical notes on imports and AST (updated May 2026)

- Top-level `import` statements are parsed into the AST root as `Program.imports` where each entry contains:
    - `raw_path`: the literal path or module name as written in source
    - `alias`: the local name to reference the module
    - `module_name`: canonical resolved module name (for builtins or resolved file path)
- Builtins (`std`, `math`, `tensor`, `nn`, `optim`, `data`, `parallel`) are resolved without file paths and provided via `io::BuiltinRegistry::with_builtins()`.
- File imports are resolved using `io::ImportResolver` which parses, semantically validates, and registers the imported module's exports into the `BuiltinRegistry` so that `IRBuilder` and the handlers can consume them.

### Notes for module implementers

 - If your module needs custom lowering into IR, implement an `io::ModuleHandler` in `compiler/ir/ir_modules/` (see `tensor_handler.h` for example) and register it in `ModuleHandlerRegistry::with_builtins()`.
 - Add any `.cpp` implementation files to `CMakeLists.txt` under `CORE_SOURCES` so they are compiled into `tensorc_lib`.
 - Keep handler logic limited to `lower_call()` — this keeps `IRBuilder` generic and avoids edits to core lowering.


---

## Built-in Modules Reference

### std - Standard Library

**Import:** `import std;`

#### I/O Functions

```tensorc
std.print(value: any) -> void
// Outputs value to stdout without newline
std.println(value: any) -> void
// Outputs value to stdout with newline
std.eprint(value: any) -> void
// Outputs value to stderr without newline
std.eprintln(value: any) -> void
// Outputs value to stderr with newline

std.read_line() -> str
// Read a line from stdin
```

**Example:**
```tensorc
let name: str = std.read_line();
std.println("Hello, " + name + "!");
```

#### Assertions & Debugging

```tensorc
std.assert(condition: bool) -> void
// Panic if condition is false

std.assert_eq(expected: any, actual: any) -> void
// Panic if not equal

std.type_of(value: any) -> str
// Get runtime type name

std.panic(message: str) -> void
// Immediately exit with error
```

**Example:**
```tensorc
let x: i32 = 5;
std.assert(x > 0);                    // OK
std.assert_eq(x, 5);                  // OK
std.println(std.type_of(x));          // "i32"
std.panic("This should not happen");  // Exit
```

#### Program Control

```tensorc
std.exit(code: i32) -> void
// Exit program with code
```

#### Collections

```tensorc
std.len(arr: Array<T>) -> i64
// Get array length

std.range(start: i64, end: i64) -> Array<i64>
// Generate array [start, start+1, ..., end-1]
```

**Example:**
```tensorc
let numbers: Array<i64> = std.range(0, 10);
let count: i64 = std.len(numbers);  // 10
```

---

### math - Mathematical Functions

**Import:** `import math;`

#### Unary Functions

```tensorc
// Roots and exponents
math.sqrt(x: f32) -> f32
math.cbrt(x: f32) -> f32
math.exp(x: f32) -> f32      // e^x
math.exp2(x: f32) -> f32     // 2^x
math.log(x: f32) -> f32      // natural log
math.log2(x: f32) -> f32
math.log10(x: f32) -> f32

// Basic
math.abs(x: f32) -> f32
math.sign(x: f32) -> f32

// Trigonometric
math.sin(x: f32) -> f32
math.cos(x: f32) -> f32
math.tan(x: f32) -> f32
math.asin(x: f32) -> f32
math.acos(x: f32) -> f32
math.atan(x: f32) -> f32

// Hyperbolic
math.sinh(x: f32) -> f32
math.cosh(x: f32) -> f32
math.tanh(x: f32) -> f32

// Rounding
math.floor(x: f32) -> f32
math.ceil(x: f32) -> f32
math.round(x: f32) -> f32
math.trunc(x: f32) -> f32
```

#### Binary Functions

```tensorc
math.pow(base: f32, exp: f32) -> f32
math.atan2(y: f32, x: f32) -> f32
math.hypot(x: f32, y: f32) -> f32    // sqrt(x² + y²)
math.fmod(x: f32, y: f32) -> f32     // Floating-point remainder
math.max(x: f32, y: f32) -> f32
math.min(x: f32, y: f32) -> f32
math.clamp(x: f32, min: f32, max: f32) -> f32
```

#### Constants

```tensorc
math.pi: f32       // 3.14159...
math.e: f32        // 2.71828...
math.inf: f32      // Positive infinity
math.nan: f32      // Not a number
math.epsilon: f32  // Machine epsilon (~1.2e-7)
```

**Example:**
```tensorc
import math;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + math.exp(-x))
}

fn gaussian(x: f32, mu: f32, sigma: f32) -> f32 {
    let coeff: f32 = 1.0 / (sigma * math.sqrt(2.0 * math.pi));
    let exp_term: f32 = -0.5 * ((x - mu) / sigma) * ((x - mu) / sigma);
    coeff * math.exp(exp_term)
}
```

---

### tensor - Tensor Operations

**Import:** `import tensor;` or `import tensor as ts;`

#### Creation

```tensorc
tensor.zeros(shape: Array<i64>) -> Tensor<f32>
// Create tensor filled with 0
tensor.ones(shape: Array<i64>) -> Tensor<f32>
// Create tensor filled with 1
tensor.full(shape: Array<i64>, value: f32) -> Tensor<f32>
// Create tensor filled with value
tensor.eye(shape: Array<i64>) -> Tensor<f32>
// Create identity-like tensor

tensor.arange(start: i64, end: i64, step: i64) -> Tensor<f32>
// Create range tensor
tensor.linspace(start: f32, end: f32, steps: i64) -> Tensor<f32>
// Create evenly-spaced tensor

tensor.rand(shape: Array<i64>) -> Tensor<f32>
// Uniform random [0, 1)
tensor.randn(shape: Array<i64>) -> Tensor<f32>
// Normal distribution
tensor.randint(low: i64, high: i64, shape: Array<i64>) -> Tensor<i64>
// Random integers

tensor.from_list(values: Array<f32>) -> Tensor<f32>
// Create tensor from list
```

#### Shape Operations

```tensorc
// Methods (not functions):
tensor_value.shape() -> Array<i64>
tensor_value.rank() -> i32
tensor_value.size() -> i32
tensor_value.dtype() -> str

tensor_value.reshape(shape: Array<i64>) -> Tensor<f32>
tensor_value.flatten() -> Tensor<f32>
tensor_value.transpose() -> Tensor<f32>
tensor_value.permute(dims: Array<i32>) -> Tensor<f32>
```

#### Arithmetic

```tensorc
tensor.dot(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
// Dot product / matrix multiply

tensor.matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
// Matrix multiplication

tensor.outer(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
// Outer product

tensor.add(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
tensor.subtract(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
tensor.multiply(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
tensor.divide(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
```

#### Reduction

```tensorc
// Methods:
tensor_value.sum() -> f32
tensor_value.mean() -> f32
tensor_value.min() -> f32
tensor_value.max() -> f32
tensor_value.prod() -> f32
tensor_value.norm() -> f32
```

#### Linear Algebra

```tensorc
tensor.det(a: Tensor<f32>) -> f32
// Determinant

tensor.trace(a: Tensor<f32>) -> f32
// Trace (sum of diagonals)

tensor.inv(a: Tensor<f32>) -> Tensor<f32>
// Matrix inverse

tensor.solve(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32>
// Solve Ax = b
```

#### Gradients (Autodiff)

```tensorc
// On Tensor<f32> with requires_grad=true:
tensor_value.requires_grad() -> bool
tensor_value.backward() -> void
tensor_value.grad() -> Tensor<f32>
tensor_value.zero_grad() -> void
```

**Example:**
```tensorc
import tensor as ts;

fn linear_regression(X: Tensor<f32>, y: Tensor<f32>) -> Tensor<f32> {
    // X: (n_samples, n_features)
    // y: (n_samples,)
    // Returns: weights (n_features,)
    
    let XtX: Tensor<f32> = ts.matmul(ts.transpose(X), X);
    let Xty: Tensor<f32> = ts.matmul(ts.transpose(X), y);
    let weights: Tensor<f32> = ts.solve(XtX, Xty);
    weights
}
```

---

### nn - Neural Network Layers

**Import:** `import nn;`

#### Layer Construction

```tensorc
// Create layers (returns layer objects)
nn.Linear(in_features: i32, out_features: i32) -> Layer
nn.Conv2d(in_channels: i32, out_channels: i32, kernel_size: i32, 
          stride: i32, padding: i32) -> Layer
nn.BatchNorm(num_features: i32) -> Layer
nn.Dropout(p: f32) -> Layer
nn.LayerNorm(shape: Array<i32>) -> Layer

// Activation functions
nn.ReLU() -> Layer
nn.Sigmoid() -> Layer
nn.Tanh() -> Layer
nn.Softmax(dim: i32) -> Layer
nn.LeakyReLU(alpha: f32) -> Layer
nn.GELU() -> Layer
```

#### Layer Operations

```tensorc
// Forward pass
layer.forward(input: Tensor<f32>) -> Tensor<f32>

// Gradient computation
layer.backward(grad_output: Tensor<f32>) -> Tensor<f32>

// Parameter access
layer.parameters() -> Array<Tensor<f32>>
layer.get_param(name: str) -> Tensor<f32>
layer.set_param(name: str, value: Tensor<f32>) -> void

// Training mode
layer.train() -> void
layer.eval() -> void
```

**Example:**
```tensorc
import nn;
import tensor as ts;

fn create_mlp(input_size: i32, hidden_size: i32, output_size: i32) {
    let layers: Array<Layer> = [];
    layers.push(nn.Linear(input_size, hidden_size));
    layers.push(nn.ReLU());
    layers.push(nn.Linear(hidden_size, output_size));
    layers.push(nn.Softmax(1));
    layers
}

fn forward_pass(layers: Array<Layer>, x: Tensor<f32>) -> Tensor<f32> {
    let mut out: Tensor<f32> = x;
    for layer in layers {
        out = layer.forward(out);
    }
    out
}
```

---

### optim - Optimization Algorithms

**Import:** `import optim;`

#### Optimizer Creation

```tensorc
optim.SGD(params: Array<Tensor<f32>>, lr: f32) -> Optimizer
optim.SGDMomentum(params: Array<Tensor<f32>>, lr: f32, momentum: f32) -> Optimizer
optim.Adam(params: Array<Tensor<f32>>, lr: f32, betas: [f32; 2]) -> Optimizer
optim.RMSprop(params: Array<Tensor<f32>>, lr: f32, alpha: f32) -> Optimizer
optim.AdaGrad(params: Array<Tensor<f32>>, lr: f32) -> Optimizer
```

#### Optimizer Operations

```tensorc
optimizer.step() -> void
// Update parameters

optimizer.zero_grad() -> void
// Reset gradients

optimizer.get_lr() -> f32
optimizer.set_lr(lr: f32) -> void
```

**Example:**
```tensorc
import nn;
import optim;
import tensor as ts;

fn training_loop(model: Array<Layer>, train_data: Array<[Tensor<f32>; 2]>, 
                 num_epochs: i32) {
    let optimizer: Optimizer = optim.Adam(model_params(model), 0.001);
    
    for epoch in std.range(0, num_epochs) {
        for [x, y] in train_data {
            let pred: Tensor<f32> = forward_pass(model, x);
            let loss: f32 = compute_loss(pred, y);
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}
```

---

## Creating Custom Modules

### Current Status

**Note:** User-defined modules are not yet fully supported in the current version. This section describes the future API.

### Basic Structure

**File:** `my_project/utils.tcc`

```tensorc
// Module documentation
// This module provides utility functions for data processing

// Import dependencies
import std;
import math;

// Global constants
let EPSILON: f32 = 1e-8;

// Function definitions

/// Normalize vector to unit length
fn normalize(vec: Tensor<f32>) -> Tensor<f32> {
    let norm: f32 = vec.norm();
    if norm > EPSILON {
        vec / norm
    } else {
        vec
    }
}

/// Compute moving average
fn moving_average(data: Array<f32>, window: i32) -> Array<f32> {
    let mut result: Array<f32> = [];
    for i in std.range(0, std.len(data) - window) {
        let mut sum: f32 = 0.0;
        for j in std.range(0, window) {
            sum = sum + data[i + j];
        }
        result.push(sum / window as f32);
    }
    result
}

/// Compute standard deviation
fn std_dev(data: Array<f32>) -> f32 {
    let n: i32 = std.len(data) as i32;
    let mean: f32 = sum_array(data) / n as f32;
    
    let mut variance: f32 = 0.0;
    for val in data {
        variance = variance + (val - mean) * (val - mean);
    }
    variance = variance / n as f32;
    
    math.sqrt(variance)
}

// Helper (not exported by default, but accessible within module)
fn sum_array(arr: Array<f32>) -> f32 {
    let mut total: f32 = 0.0;
    for val in arr {
        total = total + val;
    }
    total
}
```

### Module Organization

For larger modules, organize into subdirectories:

```
my_project/
├── math_utils/
│   ├── core.tcc           // Basic math functions
│   ├── matrix_ops.tcc     // Matrix operations
│   ├── stats.tcc          // Statistical functions
│   └── index.tcc          // Re-exports from submodules
├── data_utils/
│   ├── loading.tcc        // Data loading
│   ├── preprocessing.tcc  // Data preprocessing
│   └── index.tcc
└── main.tcc
```

**File:** `my_project/math_utils/index.tcc`

```tensorc
// Re-export submodules
import "./core" as core;
import "./matrix_ops" as matrix;
import "./stats" as stats;

// Make available to importers
// Users: import "./math_utils" as math_util
```

### Using Custom Modules

**File:** `my_project/main.tcc`

```tensorc
import "./utils" as utils;
import "./math_utils" as math_util;
import std;
import tensor as ts;

fn main() {
    // Use custom module functions
    let vec: Tensor<f32> = ts.rand([3, 3]);
    let norm_vec: Tensor<f32> = utils.normalize(vec);
    std.println(norm_vec);
    
    let data: Array<f32> = [1.0, 2.0, 3.0, 4.0, 5.0];
    let std_dev: f32 = utils.std_dev(data);
    std.println(std_dev);
}
```

---

## Module Best Practices

### 1. Naming Conventions

**Module Names:** Use lowercase with underscores
```tensorc
import "./my_math_utils" as my_math;  // Good
import "./MyMathUtils" as MyMath;     // Avoid
```

**Function Names:** Use snake_case
```tensorc
fn compute_average(values: Array<f32>) -> f32 { ... }  // Good
fn computeAverage(values: Array<f32>) -> f32 { ... }   // Avoid
```

**Type/Module Exports:** Use PascalCase for structs (when available)
```tensorc
struct DataPoint { ... }  // Good
struct data_point { ... } // Avoid
```

### 2. Documentation

**Always document public functions:**

```tensorc
/// Computes element-wise maximum of two tensors.
///
/// # Arguments
/// * `a` - First tensor (shape: [N, M])
/// * `b` - Second tensor (shape: [N, M])
///
/// # Returns
/// A tensor of shape [N, M] where each element is max(a[i,j], b[i,j])
///
/// # Example
/// ```
/// let x: Tensor<f32> = ts.ones([3, 3]);
/// let y: Tensor<f32> = ts.full([3, 3], 2.0);
/// let z: Tensor<f32> = element_wise_max(x, y);  // All 2.0
/// ```
fn element_wise_max(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
    // Implementation
}
```

### 3. Error Handling

**Validate inputs:**

```tensorc
fn matrix_multiply(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
    let shape_a: Array<i64> = a.shape();
    let shape_b: Array<i64> = b.shape();
    
    // Verify dimensions
    if shape_a[1] != shape_b[0] {
        std.panic("Dimension mismatch in matrix multiply");
    }
    
    // Perform operation
    ts.matmul(a, b)
}
```

### 4. Performance Considerations

**Minimize copies:**
```tensorc
// Bad: Creates intermediate copy
fn bad_scaling(x: Tensor<f32>, factor: f32) -> Tensor<f32> {
    let scaled: Tensor<f32> = ts.multiply(x, factor);
    scaled.flatten()
}

// Better: Chain operations
fn good_scaling(x: Tensor<f32>, factor: f32) -> Tensor<f32> {
    ts.multiply(x, factor).flatten()
}
```

**Reuse tensors when possible:**
```tensorc
fn batch_normalize(batch: Array<Tensor<f32>>) -> Array<Tensor<f32>> {
    let mut results: Array<Tensor<f32>> = [];
    let batch_size: i32 = std.len(batch) as i32;
    
    // Compute statistics once
    let mean: Tensor<f32> = compute_batch_mean(batch);
    let std_dev: Tensor<f32> = compute_batch_std(batch);
    
    // Apply to all samples
    for sample in batch {
        let norm: Tensor<f32> = (sample - mean) / std_dev;
        results.push(norm);
    }
    
    results
}
```

### 5. Version Compatibility

Mark module versions in documentation:

```tensorc
// my_utils.tcc
// Version: 1.0.0
// Compatible with: TensorC >= 0.1.0
// Last updated: 2026-05-23

import std;
import tensor;
```

---

## Extending Built-in Modules

### Proposal: Module Extension Points

For future versions, consider standard extension mechanisms:

#### 1. Custom Operations

```tensorc
// Allow registering custom tensor operations
register_tensor_op("my_conv_fused", my_conv_fused_impl);
```

#### 2. Custom Types

```tensorc
// Extend type system with custom types
register_type("Graph", GraphType);
register_type("Mesh", MeshType);
```

#### 3. Compile-Time Plugins

```tensorc
// Execute code at compile time
#[compile_time]
fn precompute_weights() {
    // Runs during compilation
}
```

---

## Publishing Modules

### Proposed Publishing Workflow (Future)

#### 1. Prepare Module

```
my_module/
├── src/
│   ├── lib.tcc          // Main module file
│   ├── util.tcc         // Utilities
│   └── tests.tcc        // Tests
├── examples/
│   ├── basic.tcc
│   └── advanced.tcc
├── tensorc.toml         # Module metadata
└── README.md
```

**File:** `tensorc.toml`

```toml
[package]
name = "my-math-utils"
version = "1.0.0"
authors = ["Your Name <you@example.com>"]
description = "Mathematical utilities for TensorC"

[dependencies]
tensorc = ">= 0.1.0"
std = "*"
math = "*"

[module]
entry = "src/lib.tcc"
docs = true
```

#### 2. Publish to Registry

```bash
tensorc publish --token <api-key>
```

#### 3. Users Install

```bash
tensorc add my-math-utils
```

**File:** `tensorc.toml` (in user project)

```toml
[dependencies]
my-math-utils = "1.0"
```

**File:** User code

```tensorc
import "my-math-utils" as math_utils;

fn main() {
    let result: f32 = math_utils.compute_something();
}
```

---

## Summary

| Aspect | Current | Future |
|--------|---------|--------|
| Built-in modules | ✅ Yes | ✅ Expanded |
| User modules | ❌ No | ✅ Yes |
| Module docs | ⚠️ Partial | ✅ Complete |
| Custom handlers | ❌ No | ✅ Yes |
| Package registry | ❌ No | ✅ Planned |
| Version management | ❌ No | ✅ Planned |

TensorC's module system is evolving from built-in-only to a full plugin ecosystem. This document describes both the current capabilities and the roadmap for extensibility.
