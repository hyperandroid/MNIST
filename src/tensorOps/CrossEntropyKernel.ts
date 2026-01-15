import {Kernel, ceilDiv} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";

/**
 * Cross Entropy Loss kernel.
 * Computes: -sum(labels * log(predictions + epsilon)) per row.
 *
 * Input predictions should be softmax output [M, N].
 * Input labels should be one-hot encoded [M, N].
 * Output is per-sample loss [M, 1].
 */
export class CrossEntropyKernel extends Kernel {

	static readonly CROSS_ENTROPY_OUTPUT = "xentropy_out";
	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
	) {
		super(device, CrossEntropyKernel.xentropyWGSL);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		predictions: Tensor,
		labels: Tensor,
		out?: Tensor,
	): Tensor {
		if (predictions.shape.length !== 2) {
			throw new Error("CrossEntropy: predictions must be 2D tensor");
		}
		if (labels.shape.length !== 2) {
			throw new Error("CrossEntropy: labels must be 2D tensor");
		}
		if (predictions.shape[0] !== labels.shape[0] || predictions.shape[1] !== labels.shape[1]) {
			throw new Error("CrossEntropy: predictions and labels must have same shape");
		}

		const M = predictions.shape[0];
		const N = predictions.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			CrossEntropyKernel.CROSS_ENTROPY_OUTPUT,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, 1],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: predictions.buffer}},
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

	static xentropyWGSL = `
		// Cross Entropy Loss: L = -sum(y_true * log(y_pred + epsilon))
		// One thread per row (sample).

		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> predictions : array<f32>;
		@group(0) @binding(1) var<storage, read> labels : array<f32>;
		@group(0) @binding(2) var<storage, read_write> loss : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		const EPSILON : f32 = 1e-7;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let row = gid.x;
			if (row >= params.M) {
				return;
			}

			let N = params.N;
			let base = row * N;

			var sum = 0.0;
			for (var i = 0u; i < N; i = i + 1u) {
				let pred = predictions[base + i];
				let label = labels[base + i];
				sum = sum + label * log(pred + EPSILON);
			}

			loss[row] = -sum;
		}
	`;
}
