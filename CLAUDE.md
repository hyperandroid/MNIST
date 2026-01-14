# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
npm run dev    # Start Vite dev server
```

## Architecture

This is a WebGPU-based tensor/matrix computation project for MNIST, built with TypeScript and Vite.

### Key Components

- **src/Tensor.ts**: Core Tensor class that wraps `Float32Array` with shape metadata. Supports 1D and 2D arrays with row-major storage layout.

- **index.html**: Contains a WebGPU compute shader demo implementing tiled 16x16 matrix multiplication in WGSL. The shader uses workgroup shared memory for efficient tiled matmul (C = A × B).

### WebGPU Compute Pattern

The matmul shader follows the standard WebGPU compute pattern:
1. Storage buffers for input matrices A, B and output C
2. Uniform buffer for matrix dimensions (M, N, K)
3. Workgroup-local tile arrays (16×16) with barrier synchronization
4. Dispatch workgroups based on output dimensions: `(ceil(N/16), ceil(M/16), 1)`
