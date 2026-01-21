import {Kernel, ceilDiv} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import type {KernelRegistry} from "./KernelRegistry";
import {SoftmaxBackward} from "../../autograd/backward/SoftmaxBackward";

/**
 * Per sample softmax.
 * Expected to be used with small Tensors (e.g. MNIST N=10)
 *
 * This implementation is not optimized for large tensors.
 */
export class SoftmaxKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, SoftmaxKernel.softmaxWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		t0: Tensor,
		out?: Tensor,
	): Tensor {
		if (t0.shape.length !== 2) {
			throw new Error("Softmax: expected 2D tensor");
		}

		const M = t0.shape[0];
		const N = t0.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: t0.buffer}},
				{binding: 1, resource: {buffer: out.buffer}},
				{binding: 2, resource: {buffer: this.paramsBuf}},
			],
		});

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(ceilDiv(M, 256), 1, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);

		// Autograd: track computation graph
		// Save input for backward. Note: Softmax backward is typically
		// handled via combined SoftmaxCrossEntropy for numerical stability.
		if (t0.requiresGradient) {
			out.requiresGradient = true;
			out.parents = [t0];
			// Store both input and output for flexible backward computation
			out.gradFn = new SoftmaxBackward([t0, out], this.kr!);
		}

		return out;
	}

	static softmaxWGSL = `
		// Softmax: exp(x_i - max) / sum(exp(x_j - max))
		// Applied per row for numerical stability.

		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read_write> B : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let row = gid.x;
			if (row >= params.M) {
				return;
			}

			let N = params.N;
			let base = row * N;

			// Find max for numerical stability
			var maxVal = A[base];
			for (var i = 1u; i < N; i = i + 1u) {
				maxVal = max(maxVal, A[base + i]);
			}

			// Compute exp(x - max) and sum
			var sum = 0.0;
			for (var i = 0u; i < N; i = i + 1u) {
				let e = exp(A[base + i] - maxVal);
				B[base + i] = e;
				sum = sum + e;
			}

			// Normalize
			let invSum = 1.0 / sum;
			for (var i = 0u; i < N; i = i + 1u) {
				B[base + i] = B[base + i] * invSum;
			}
		}
	`;
}
