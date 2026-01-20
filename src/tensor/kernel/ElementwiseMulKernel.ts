import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Element-wise multiplication (Hadamard product).
 *
 * Input A: [M, N]
 * Input B: [M, N]
 * Output:  [M, N] where output[i,j] = A[i,j] * B[i,j]
 *
 * Used for gradient masking (ReLU backward, Dropout backward).
 */
export class ElementwiseMulKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, ElementwiseMulKernel.elemMulWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		a: Tensor,
		b: Tensor,
		out?: Tensor,
	): Tensor {
		if (a.shape.length !== 2 || b.shape.length !== 2) {
			throw new Error("ElementwiseMul: inputs must be 2D tensors");
		}

		if (a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1]) {
			throw new Error("ElementwiseMul: input shapes must match");
		}

		const M = a.shape[0];
		const N = a.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			`${a.name}_${b.name}_mul`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: a.buffer}},
				{binding: 1, resource: {buffer: b.buffer}},
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

	static elemMulWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read> B : array<f32>;
		@group(0) @binding(2) var<storage, read_write> C : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			C[idx] = A[idx] * B[idx];
		}
	`;
}
