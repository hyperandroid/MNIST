import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Combined Softmax + CrossEntropy backward kernel.
 *
 * Computes: gradLogits = softmax - labels
 *
 * This is the gradient of CrossEntropy(Softmax(logits), labels) w.r.t. logits.
 * Numerically stable and efficient as a single operation.
 *
 * Input softmax: [M, N] (output of softmax forward)
 * Input labels:  [M, N] (one-hot encoded)
 * Output:        [M, N]
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
		softmax: Tensor,
		labels: Tensor,
		out?: Tensor,
	): Tensor {
		if (softmax.shape.length !== 2 || labels.shape.length !== 2) {
			throw new Error("SoftmaxCEBackward: inputs must be 2D tensors");
		}

		if (softmax.shape[0] !== labels.shape[0] ||
			softmax.shape[1] !== labels.shape[1]) {
			throw new Error("SoftmaxCEBackward: shapes must match");
		}

		const M = softmax.shape[0];
		const N = softmax.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			`${softmax.name}_ce_bwd`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: softmax.buffer}},
				{binding: 1, resource: {buffer: labels.buffer}},
				{binding: 2, resource: {buffer: out.buffer}},
				{binding: 3, resource: {buffer: this.paramsBuf}},
			],
		});

		const wgX = ceilDiv(N, 16);
		const wgY = ceilDiv(M, 16);

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(wgX, wgY, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);

		return out;
	}

	static softmaxCEBackwardWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> softmax : array<f32>;
		@group(0) @binding(1) var<storage, read> labels : array<f32>;
		@group(0) @binding(2) var<storage, read_write> gradLogits : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			// gradient = softmax - label (for each element)
			gradLogits[idx] = softmax[idx] - labels[idx];
		}
	`;
}
