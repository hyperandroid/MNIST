import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Combined Softmax + CrossEntropy backward kernel.
 *
 * Computes: gradLogits = softmax(logits) - labels
 *
 * This is the gradient of CrossEntropy(Softmax(logits), labels) w.r.t. logits.
 * Numerically stable and efficient as a single operation.
 *
 * Input logits: [M, N] (raw logits, NOT softmax output)
 * Input labels: [M, N] (one-hot encoded)
 * Output:       [M, N]
 */
export class SoftmaxCEBackwardKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, SoftmaxCEBackwardKernel.softmaxCEBackwardWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		logits: Tensor,
		labels: Tensor,
		out?: Tensor,
	): Tensor {
		if (logits.shape.length !== 2 || labels.shape.length !== 2) {
			throw new Error("SoftmaxCEBackward: inputs must be 2D tensors");
		}

		if (logits.shape[0] !== labels.shape[0] ||
			logits.shape[1] !== labels.shape[1]) {
			throw new Error("SoftmaxCEBackward: shapes must match");
		}

		const M = logits.shape[0];
		const N = logits.shape[1];

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
				{binding: 0, resource: {buffer: logits.buffer}},
				{binding: 1, resource: {buffer: labels.buffer}},
				{binding: 2, resource: {buffer: out.buffer}},
				{binding: 3, resource: {buffer: this.paramsBuf}},
			],
		});

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(ceilDiv(M, 256), 1, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);

		return out;
	}

	static softmaxCEBackwardWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> logits : array<f32>;
		@group(0) @binding(1) var<storage, read> labels : array<f32>;
		@group(0) @binding(2) var<storage, read_write> gradLogits : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let row = gid.x;
			if (row >= params.M) {
				return;
			}

			let N = params.N;
			let base = row * N;

			// 1) Find max for numerical stability
			var maxVal = logits[base];
			for (var i = 1u; i < N; i = i + 1u) {
				maxVal = max(maxVal, logits[base + i]);
			}

			// 2) Compute exp(x - max) and sum
			var sum = 0.0;
			for (var i = 0u; i < N; i = i + 1u) {
				let e = exp(logits[base + i] - maxVal);
				gradLogits[base + i] = e;
				sum = sum + e;
			}

			// 3) Normalize to get softmax, then subtract labels
			let invSum = 1.0 / sum;
			for (var i = 0u; i < N; i = i + 1u) {
				let softmax_i = gradLogits[base + i] * invSum;
				gradLogits[base + i] = softmax_i - labels[base + i];
			}
		}
	`;
}
