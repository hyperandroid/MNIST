import {Kernel, ceilDiv} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Softmax backward kernel.
 *
 * Computes: dX = S * (dS - sum(dS * S, axis=1, keepdims=True))
 *
 * where S = softmax output, dS = gradient of loss w.r.t. softmax output
 *
 * Input gradOutput: [M, N] (dL/dSoftmax)
 * Input softmaxOut: [M, N] (softmax output from forward)
 * Output gradInput: [M, N] (dL/dX)
 */
export class SoftmaxBackwardKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, SoftmaxBackwardKernel.softmaxBackwardWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		gradOutput: Tensor,
		softmaxOut: Tensor,
		out?: Tensor,
	): Tensor {
		if (gradOutput.shape.length !== 2 || softmaxOut.shape.length !== 2) {
			throw new Error("SoftmaxBackward: inputs must be 2D tensors");
		}

		if (gradOutput.shape[0] !== softmaxOut.shape[0] ||
			gradOutput.shape[1] !== softmaxOut.shape[1]) {
			throw new Error("SoftmaxBackward: shapes must match");
		}

		const M = gradOutput.shape[0];
		const N = gradOutput.shape[1];

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
				{binding: 0, resource: {buffer: gradOutput.buffer}},
				{binding: 1, resource: {buffer: softmaxOut.buffer}},
				{binding: 2, resource: {buffer: out.buffer}},
				{binding: 3, resource: {buffer: this.paramsBuf}},
			],
		});

		// One thread per row
		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(ceilDiv(M, 256), 1, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);

		return out;
	}

	static softmaxBackwardWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
		@group(0) @binding(1) var<storage, read> softmaxOut : array<f32>;
		@group(0) @binding(2) var<storage, read_write> gradInput : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let row = gid.x;
			if (row >= params.M) {
				return;
			}

			let N = params.N;
			let base = row * N;

			// Compute dot = sum(dS * S) for this row
			var dot = 0.0;
			for (var i = 0u; i < N; i = i + 1u) {
				dot = dot + gradOutput[base + i] * softmaxOut[base + i];
			}

			// Compute gradInput = S * (dS - dot)
			for (var i = 0u; i < N; i = i + 1u) {
				let idx = base + i;
				gradInput[idx] = softmaxOut[idx] * (gradOutput[idx] - dot);
			}
		}
	`;
}
