# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
npm run dev    # Start Vite dev server
```

## Project Structure

```
src/
├── GPUEnv.ts                    # WebGPU device singleton
├── TrainModel.ts                # Entry point
├── model/
│   ├── Model.ts                 # Model interface
│   └── Datasource.ts            # Datasource interface
├── MNIST/
│   └── MNISTDatasource.ts       # MNIST data loading
├── layer/
│   ├── Layer.ts                 # Base layer interface
│   ├── Linear.ts                # Fully connected layer
│   ├── Sequential.ts            # Layer container
│   ├── ReLU.ts                  # ReLU activation layer
│   └── Dropout.ts               # Dropout regularization layer
├── math/
│   └── Utils.ts                 # Weight initializers (heNormal, heUniform)
└── tensor/
    ├── Tensor.ts                # GPU-backed tensor
    ├── TensorManager.ts         # Buffer lifecycle management
    └── kernel/
        ├── Kernel.ts            # Base kernel class
        ├── KernelRegistry.ts    # Kernel instantiation
        ├── MatMulKernel.ts
        ├── MatAddKernel.ts
        ├── BiasAddKernel.ts
        ├── RELUKernel.ts
        ├── DropoutKernel.ts
        ├── SoftmaxKernel.ts
        └── CrossEntropyKernel.ts
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

- **MatMulKernel**: Tiled 16×16 matrix multiplication with shared memory
- **MatAddKernel**: Element-wise addition of two 2D tensors with matching shapes
- **BiasAddKernel**: Broadcasts 1D bias [N] across rows of 2D input [M,N]
- **RELUKernel**: Element-wise ReLU activation
- **DropoutKernel**: Element-wise multiplication with pre-computed mask
- **SoftmaxKernel**: Per-row softmax with numerical stability (optimized for small N like MNIST's 10 classes)
- **CrossEntropyKernel**: Per-sample cross-entropy loss. Takes predictions [M,N] and one-hot labels [M,N], outputs loss [M,1]

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
