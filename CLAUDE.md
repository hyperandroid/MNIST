# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
npm run dev    # Start Vite dev server
```

## Project Structure

```
src/
├── GPUEnv.ts                           # WebGPU device singleton
├── main.ts                             # Entry point
├── Trainer.ts                          # Training loop orchestrator
├── Tester.ts                           # Test/evaluation harness
├── model/
│   └── Datasource.ts                   # Datasource interface
├── MNIST/
│   ├── MNIST.ts                        # MNIST model definition
│   └── MNISTDatasource.ts              # MNIST data loading
├── layer/
│   ├── Layer.ts                        # Base layer interface
│   ├── Linear.ts                       # Fully connected layer
│   ├── Sequential.ts                   # Layer container
│   ├── ReLU.ts                         # ReLU activation layer
│   └── Dropout.ts                      # Dropout regularization layer
├── math/
│   └── Utils.ts                        # Weight initializers (heNormal, heUniform)
├── autograd/
│   ├── GradientFunction.ts             # Interface for backward functions
│   ├── BackwardPass.ts                 # Reverse topological sort for backprop
│   └── backward/
│       ├── MatMulBackward.ts
│       ├── BiasAddBackward.ts
│       ├── ReLUBackward.ts
│       ├── DropoutBackward.ts
│       ├── MatAddBackward.ts
│       ├── SoftmaxBackward.ts
│       └── SoftmaxCrossEntropyBackward.ts
├── optimizer/
│   ├── Optimizer.ts                    # Optimizer interface with LR scheduling
│   └── SGD.ts                          # SGD with learning rate scheduling
└── tensor/
    ├── Tensor.ts                       # GPU-backed tensor with autograd support
    ├── TensorManager.ts                # Buffer lifecycle management
    └── kernel/
        ├── Kernel.ts                   # Base kernel class
        ├── KernelRegistry.ts           # Kernel instantiation
        ├── MatMulKernel.ts
        ├── MatAddKernel.ts
        ├── BiasAddKernel.ts
        ├── RELUKernel.ts
        ├── DropoutKernel.ts
        ├── SoftmaxKernel.ts
        ├── CrossEntropyKernel.ts
        ├── TransposeKernel.ts          # Autograd support
        ├── SumReduceKernel.ts          # Autograd support
        ├── ElementwiseMulKernel.ts     # Autograd support
        ├── ReLUBackwardKernel.ts       # Autograd support
        ├── SoftmaxBackwardKernel.ts    # Autograd support
        ├── SoftmaxCEBackwardKernel.ts  # Autograd support
        ├── ScalarMulKernel.ts          # Optimizer support
        ├── InplaceAddKernel.ts         # Optimizer support
        └── SumAllKernel.ts             # Optimizer support
```

## Architecture

WebGPU-based tensor computation library for MNIST, built with TypeScript and Vite.

### Core Components

- **GPUEnv** (`src/GPUEnv.ts`): Singleton that initializes WebGPU device. Throws on failure. Must call `await GPUEnv.init()` before using `GPUEnv.device`.

- **Tensor** (`src/tensor/Tensor.ts`): Wraps `GPUBuffer` with shape metadata. Row-major storage. Size computed via `shape.reduce((a, b) => a * b, 1)`.

- **TensorManager** (`src/tensor/TensorManager.ts`): Manages GPU buffer lifecycle. Handles buffer reuse, 256-byte alignment, readback buffers, and deferred destruction. Use `getTensorBuffer()` to create/reuse tensors by name.

- **KernelRegistry** (`src/tensor/kernel/KernelRegistry.ts`): Central registry for GPU kernels. Instantiates all kernels with shared device and TensorManager.

### Kernel Pattern

All kernels extend `Kernel` base class (`src/tensor/kernel/Kernel.ts`) which creates the shader module and compute pipeline.

Kernels follow a functional pattern—they return tensors, enabling composition:
```typescript
const out = kernelRegistry.relu.run(
    kernelRegistry.matmul.run(t0, t1)
);
```

Each kernel:
1. Validates input tensor shapes (must be 2D)
2. Auto-creates output buffer if not provided (via TensorManager)
3. Writes params to uniform buffer
4. Creates bind group, dispatches compute, submits to queue
5. Returns output tensor

### Available Kernels

**Forward Kernels:**
- **MatMulKernel**: Tiled 16×16 matrix multiplication with shared memory
- **MatAddKernel**: Element-wise addition of two 2D tensors with matching shapes
- **BiasAddKernel**: Broadcasts 1D bias [N] across rows of 2D input [M,N]
- **RELUKernel**: Element-wise ReLU activation
- **DropoutKernel**: Element-wise multiplication with pre-computed mask
- **SoftmaxKernel**: Per-row softmax with numerical stability (optimized for small N like MNIST's 10 classes)
- **CrossEntropyKernel**: Per-sample cross-entropy loss. Takes predictions [M,N] and one-hot labels [M,N], outputs loss [M,1]

**Autograd Support Kernels:**
- **TransposeKernel**: Matrix transpose [M,N] → [N,M]
- **SumReduceKernel**: Sum along axis 0, [M,N] → [1,N]
- **ElementwiseMulKernel**: Hadamard product of two tensors
- **ReLUBackwardKernel**: `dX = dY * (X > 0)`
- **SoftmaxBackwardKernel**: Jacobian-vector product for softmax
- **SoftmaxCEBackwardKernel**: Combined softmax + cross-entropy backward: `dLogits = probs - labels`

**Optimizer Support Kernels:**
- **ScalarMulKernel**: `output = input * scalar` (for learning rate scaling)
- **InplaceAddKernel**: `target += source` (for parameter updates)
- **SumAllKernel**: Reduces all elements to scalar [1,1] (for loss reduction)

### WebGPU Compute Pattern

Shaders use 16×16 workgroup size (except Softmax/CrossEntropy which use 256×1). Dispatch: `(ceil(N/16), ceil(M/16), 1)`.

MatMul uses tiled algorithm with workgroup-local shared memory and barrier synchronization for coalesced memory access.

### Adding New Kernels

1. Create class extending `Kernel` in `src/tensor/kernel/`
2. Define WGSL shader as static string
3. Add shape validation, params buffer, bind group creation
4. Register in `KernelRegistry`

### Layer System

All layers implement the `Layer` interface (`src/layer/Layer.ts`):
```typescript
interface Layer {
    forward(input: Tensor): Tensor;
    backward(input: Tensor): void;
    parameters(): Tensor[];
}
```

**Available Layers:**

- **Linear**: Fully connected layer `output = input * W + b`
  - Weight layout: `[inputFeatures, outputFeatures]` (row-major)
  - Input: `[batch, inputFeatures]`, Output: `[batch, outputFeatures]`

- **ReLU**: ReLU activation as a layer wrapper

- **Dropout**: Regularization layer with inverted dropout scaling
  - Mask generated on CPU, applied on GPU

- **Sequential**: Container for stacking layers
  ```typescript
  const model = new Sequential(
      new Linear(tm, kr, {...}),
      new ReLU(tm, kr, "relu"),
      new Dropout(tm, kr, "dropout", 0.5),
  );
  const out = model.forward(input);
  ```

### Weight Initialization

`src/math/Utils.ts` provides:
- **heNormal**: He normal initialization `N(0, sqrt(2/fanIn))`
- **heUniform**: He uniform initialization `U(-limit, limit)` where `limit = sqrt(6/fanIn)`

### Autograd System

Automatic differentiation via computation graph tracking.

**Tensor Autograd Fields** (`src/tensor/Tensor.ts`):
```typescript
gradient?: Tensor;           // Accumulated gradient
gradFn?: GradientFunction;   // Backward function for this node
parents?: Tensor[];          // Input tensors that created this tensor
requiresGradient: boolean;   // Whether to track gradients
```

**GradientFunction Interface** (`src/autograd/GradientFunction.ts`):
```typescript
interface GradientFunction {
    name: string;
    savedTensors: Tensor[];           // Tensors saved for backward
    backward(gradOutput: Tensor): Tensor[];  // Compute gradients w.r.t. inputs
}
```

**Backpropagation** (`src/autograd/BackwardPass.ts`):
1. Build reverse topological order starting from loss
2. Initialize loss gradient to ones
3. Traverse in reverse order, calling `gradFn.backward()` on each node
4. Accumulate gradients in parent tensors via `inplaceAdd`

**Backward Functions** (`src/autograd/backward/`):
- **MatMulBackward**: `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`
- **BiasAddBackward**: `dX = dY`, `dBias = sum(dY, axis=0)`
- **ReLUBackward**: `dX = dY * (X > 0)`
- **DropoutBackward**: `dX = dY * mask * scale`
- **SoftmaxCrossEntropyBackward**: `dLogits = probs - labels`

### Optimizer System

**Optimizer Interface** (`src/optimizer/Optimizer.ts`):
```typescript
type LRSchedule =
    | { type: "constant" }
    | { type: "step"; factor: number; everyNSteps: number }
    | { type: "exponential"; decayRate: number }
    | { type: "cosine"; minLr: number; maxSteps: number };

interface Optimizer {
    step(batchSizeOverride?: number): void;  // Apply gradients to parameters
    zeroGrad(): void;                        // Reset all gradients to zero
    getLearningRate(): number;
    setLearningRate(lr: number): void;
    setSchedule(schedule: LRSchedule): void;
}
```

**SGD Optimizer** (`src/optimizer/SGD.ts`):
```typescript
const optimizer = new SGD(model.parameters(), learningRate, tm, kr, batchSize);
optimizer.setSchedule({ type: "cosine", minLr: 0.001, maxSteps: totalSteps });
optimizer.zeroGrad();  // Before forward pass
// ... forward, backward ...
optimizer.step(currentBatchSize);  // Update: param = param - lr * grad / batchSize
```

Features:
- Learning rate scheduling (constant, step decay, exponential decay, cosine annealing)
- Batch size normalization of gradients

### MNIST Model

**MNIST Class** (`src/MNIST/MNIST.ts`):
```typescript
class MNIST {
    readonly model: Sequential;
    constructor(tm: TensorManager, kernelRegistry: KernelRegistry, initializer = heUniform);
    async readSnapshot(): Promise<void>;  // Load pre-trained weights
    async restart(): Promise<void>;        // Reinitialize weights
}
```

The MNIST model is a 2-layer MLP: `784 → 128 (ReLU) → 10`

### Training/Testing Architecture

**Trainer** (`src/Trainer.ts`):
- Manages training loop with `requestAnimationFrame` for non-blocking UI
- Supports epoch-based training with configurable batch size
- Handles LR scheduling via cosine annealing
- State machine: `idle → training → finished` (with cancellation support)

**Tester** (`src/Tester.ts`):
- Evaluates model accuracy on test set
- Non-blocking batch processing via `requestAnimationFrame`
- Computes accuracy by comparing argmax of predictions vs labels

### Training Loop Pattern

```typescript
// Using Trainer class
const trainer = new Trainer(tm, kr, mnist, datasource, onFinished, onUpdate);
await trainer.initialize();
trainer.startTraining();

// Or manual loop
optimizer.zeroGrad();
const logits = model.forward(input, true);
const loss = kr.crossEntropy.run(logits, labelsOneHot);
computeBackwardPass(tm, kr, loss);  // Backpropagation
optimizer.step(batchSize);          // Update parameters
```
