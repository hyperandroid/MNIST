import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Multiply a tensor by a scalar.
 *
 * Input:  [M, N]
 * Scalar: f32
 * Output: [M, N] where output[i,j] = input[i,j] * scalar
 *
 * Used in optimizers for: gradient * learning_rate
 */
export class ScalarMulKernel extends Kernel {

	private readonly params = new Float32Array(4); // M, N as u32, scalar as f32, padding
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		kr: KernelRegistry,
	) {
		super(device, ScalarMulKernel.scalarMulWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		input: Tensor,
		scalar: number,
		out?: Tensor,
	): Tensor {
		if (input.shape.length !== 2) {
			throw new Error("ScalarMul: input must be 2D tensor");
		}

		const M = input.shape[0];
		const N = input.shape[1];

		// Pack params: M, N as uint32 view, then scalar as float32
		const uintView = new Uint32Array(this.params.buffer);
		uintView[0] = M;
		uintView[1] = N;
		this.params[2] = scalar;

		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: input.buffer}},
				{binding: 1, resource: {buffer: out.buffer}},
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

		return out;
	}

	static scalarMulWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		  scalar : f32,
		  _pad : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			let row = gid.y;

			if (row >= params.M || col >= params.N) {
				return;
			}

			let idx = row * params.N + col;
			output[idx] = input[idx] * params.scalar;
		}
	`;
}
