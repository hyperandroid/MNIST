import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * ReLU backward kernel.
 *
 * Computes: gradInput = gradOutput * (input > 0)
 *
 * Input gradOutput: [M, N]
 * Input savedInput: [M, N] (the original input to ReLU forward)
 * Output gradInput: [M, N]
 */
export class ReLUBackwardKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, ReLUBackwardKernel.reluBackwardWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		gradOutput: Tensor,
		savedInput: Tensor,
		out?: Tensor,
	): Tensor {
		if (gradOutput.shape.length !== 2 || savedInput.shape.length !== 2) {
			throw new Error("ReLUBackward: inputs must be 2D tensors");
		}

		if (gradOutput.shape[0] !== savedInput.shape[0] ||
			gradOutput.shape[1] !== savedInput.shape[1]) {
			throw new Error("ReLUBackward: shapes must match");
		}

		const M = gradOutput.shape[0];
		const N = gradOutput.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			`${gradOutput.name}_relu_bwd`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: gradOutput.buffer}},
				{binding: 1, resource: {buffer: savedInput.buffer}},
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

	static reluBackwardWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
		@group(0) @binding(1) var<storage, read> savedInput : array<f32>;
		@group(0) @binding(2) var<storage, read_write> gradInput : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			// gradient flows through only where input > 0
			let mask = select(0.0, 1.0, savedInput[idx] > 0.0);
			gradInput[idx] = gradOutput[idx] * mask;
		}
	`;
}
