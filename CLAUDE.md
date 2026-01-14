# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
npm run dev    # Start Vite dev server
```

## Architecture

WebGPU-based tensor computation library for MNIST, built with TypeScript and Vite.

### Core Components

- **GPUEnv**: Singleton that initializes WebGPU device. Throws on failure. Must call `await GPUEnv.init()` before using `GPUEnv.device`.

- **Tensor**: Wraps `GPUBuffer` with shape metadata. Row-major storage. Size computed via `shape.reduce((a, b) => a * b, 1)`.

- **TensorManager**: Manages GPU buffer lifecycle. Handles buffer reuse, 256-byte alignment, readback buffers, and deferred destruction. Use `getTensorBuffer()` to create/reuse tensors by name.

- **OpsRegistry**: Central registry for tensor operations. Instantiates all ops with shared device and TensorManager.

### Tensor Operations Pattern

All ops extend `TensorOp` base class which creates the shader module and compute pipeline.

Ops follow a functional pattern—they return tensors, enabling composition:
```typescript
const out = opsRegistry.relu.run(
    opsRegistry.matmul.run(t0, t1)
);
```

Each op:
1. Validates input tensor shapes (must be 2D)
2. Auto-creates output buffer if not provided (via TensorManager)
3. Writes params to uniform buffer
4. Creates bind group, dispatches compute, submits to queue
5. Returns output tensor

### WebGPU Compute Pattern

Shaders use 16×16 workgroup size. Dispatch: `(ceil(N/16), ceil(M/16), 1)`.

MatMul uses tiled algorithm with workgroup-local shared memory and barrier synchronization for coalesced memory access.

### Adding New Ops

1. Create class extending `TensorOp` in `src/tensorOps/`
2. Define WGSL shader as static string
3. Add shape validation, params buffer, bind group creation
4. Register in `OpsRegistry`
