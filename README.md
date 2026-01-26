# MNIST

A neural network trained and evaluated entirely in-browser using WebGPU compute shaders.

**Live Demo**: [hyperandroid.github.io/MNIST](https://hyperandroid.github.io/MNIST/)

## Motivation

This project could easily be done with PyTorch or TensorFlow. I built it from scratch anyway.

Over the past months, I've watched LLMs catch up to tasks I thought were uniquely human. They've become invaluable for making sense of my own thoughts. An unbelievable ability to put into words what I couldn't articulate myself. Thoughts or intuitions, nor matter what, LLM have a magic way of writting them down into words.

Turns out, I understood nothing about how they worked. Nor how they trained. Nor how they reasoned. Nor why they failed. I was using tools I couldn't see inside, and that felt asymmetrical.

In an age where knowledge has an amortized cost of zero, understanding has become the only defensible ground. Information is free. Tutorials are everywhere. But if you don't build the intuition yourself, the tools will be more like magic over time. 
And eventually your tools will even replace you. Fortunately, I don't define myself by my coding skills. One of my traits is surely to have a hacker mindset, e.g., understand problems from the ground up.

This project is my attempt to start from fundamentals. An MLP is an old technology, it dates to the 1950s. But by implementing every tensor operation, every backward pass, every gradient update from scratch, I've built something I
actually understand. It's naive. Unoptimized. Limited to two dimensions. Not production-ready. But it trains, it learns, and I know exactly why.

This is the foundation. Not the destination.   

Big thanks to [TiniTorch](https://mlsysbook.ai/tinytorch/preface.html). This project is the Typescript+WebGPU 
implementation of its [Foundation Tier](https://mlsysbook.ai/tinytorch/tiers/foundation.html).

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

- A browser with WebGPU support. This model can be trained in my iPhone 15 pro.
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
