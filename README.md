# MNIST

A neural network trained and evaluated entirely in-browser using WebGPU compute shaders.

**Live Demo**: [hyperandroid.github.io/MNIST](https://hyperandroid.github.io/MNIST/)

## Motivation

This project could be easily done using any of the existing deep learning libraries,
Pytorch, Tensorflow, etc. However, it is build from scratch.

Over the months, I have seen LLM have caught up very quickly. They are an invaluable tool for making sense of my
own thoughts and intuitions. They have an unbelievable ability to put thoughts into words and understand them.

On the contrary, I don't know anything about LLMs. I have no idea how they work internally. How they are trained.
How they are evaluated. I could be considered an absolute ML illiterate. Until now. This project is my attempt to
understand LLMs and ML from the fundamental level. By building all parts of the pipeline I have learned the fundamentals
to the point where I can train a neural network from scratch in the browser. An MLP is an old school machine learning
algorithm. IIRC it dates back to the 1950s. 

This project is the foundation for my continuous learning journey. It is a naive implementation. Unoptimized. Tensors
limited to two dimensions. Not even close to be production ready. But enough to train and solve MNIST.

Big thanks to [TiniTorch](https://mlsysbook.ai/tinytorch/preface.html). This project is the Typescript+WebGPU 
implementation of the [Foundation Tier](https://mlsysbook.ai/tinytorch/tiers/foundation.html).

## Overview

This project implements a complete deep learning pipeline for MNIST digit classification that runs entirely in the browser. All tensor operations, training, and inference are performed on the GPU using WebGPU compute shaders.

### Features

- **In-Browser Training**: Train the model from scratch on 60,000 MNIST images
- **Real-Time Inference**: Draw digits and see predictions in real-time
- **Layer Visualization**: Watch activations flow through each layer as you draw
- **Parameter Visualization**: View learned weight matrices from the pre-trained model
- **97.5% Accuracy**: Achieves high accuracy on the MNIST test set

## Architecture

The model is a Multi-Layer Perceptron (MLP):

```
Input [784] → Dense [128] → ReLU → Dense [10] → Softmax
```

- **Weight Initialization**: He uniform distribution
- **Optimizer**: Stochastic Gradient Descent (no momentum)
- **Learning Rate Schedule**: Cosine annealing

## Technical Implementation

### Tensor System

- GPU-backed tensors with automatic buffer management
- Row-major storage with 256-byte alignment for WebGPU
- Deferred buffer destruction for efficient memory reuse

### Compute Kernels

All operations implemented as WGSL compute shaders:

| Kernel | Description |
|--------|-------------|
| MatMul | Tiled 16x16 matrix multiplication with shared memory |
| BiasAdd | Broadcasts 1D bias across 2D input |
| ReLU | Element-wise activation |
| Softmax | Per-row softmax with numerical stability |
| CrossEntropy | Per-sample cross-entropy loss |
| Transpose | Matrix transpose for backprop |
| SumReduce | Reduction along axis for gradient accumulation |

### Autograd System

Automatic differentiation via computation graph:

1. Forward pass records operations and parent tensors
2. Backward pass traverses graph in reverse topological order
3. Gradients accumulated via in-place addition

## Project Structure

```
src/
├── tensor/           # GPU tensor and buffer management
│   └── kernel/       # WGSL compute shaders
├── layer/            # Neural network layers (Linear, ReLU, Dropout, Sequential)
├── autograd/         # Backward pass and gradient functions
├── optimizer/        # SGD with learning rate scheduling
├── MNIST/            # Model definition and data loading
│   └── interactive/  # UI components for the demo
└── GPUEnv.ts         # WebGPU device initialization
```

## Getting Started

### Requirements

- A browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly)
- Node.js 18+

### Development

```bash
npm install
npm run dev
```

### Build

```bash
npm run build
npm run preview   # Preview production build locally
```

### Deploy

```bash
npm run deploy    # Deploy to GitHub Pages
```

## Dataset

MNIST dataset from: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
