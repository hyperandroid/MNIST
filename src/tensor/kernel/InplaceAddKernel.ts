import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * In-place addition: target += source
 *
 * Target: [M, N] (modified in-place)
 * Source: [M, N]
 *
 * Used in optimizers for: param += update
 */
export class InplaceAddKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		kr: KernelRegistry,
	) {
		super(device, InplaceAddKernel.inplaceAddWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		target: Tensor,
		source: Tensor,
	): void {
		if (target.shape.length !== 2 || source.shape.length !== 2) {
			throw new Error("InplaceAdd: inputs must be 2D tensors");
		}

		if (target.shape[0] !== source.shape[0] || target.shape[1] !== source.shape[1]) {
			throw new Error("InplaceAdd: tensor shapes must match");
		}

		const M = target.shape[0];
		const N = target.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: target.buffer}},
				{binding: 1, resource: {buffer: source.buffer}},
				{binding: 2, resource: {buffer: this.paramsBuf}},
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
	}

	static inplaceAddWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read_write> targetTensor : array<f32>;
		@group(0) @binding(1) var<storage, read> source : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			targetTensor[idx] = targetTensor[idx] + source[idx];
		}
	`;
}
