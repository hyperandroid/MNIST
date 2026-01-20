import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Transpose a 2D matrix.
 *
 * Input:  [M, N]
 * Output: [N, M] where output[j,i] = input[i,j]
 */
export class TransposeKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, TransposeKernel.transposeWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		input: Tensor,
		out?: Tensor,
	): Tensor {
		if (input.shape.length !== 2) {
			throw new Error("Transpose: input must be 2D tensor");
		}

		const M = input.shape[0];
		const N = input.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			`${input.name}_T`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[N, M],
		);

		if (out.shape[0] !== N || out.shape[1] !== M) {
			throw new Error("Transpose: output shape must be [N, M]");
		}

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

	static transposeWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col : u32 = gid.x;  // output column = input row
			let row : u32 = gid.y;  // output row = input column

			if (row >= params.M || col >= params.N) {
				return;
			}

			// input[row, col] -> output[col, row]
			let inIdx = row * params.N + col;
			let outIdx = col * params.M + row;
			output[outIdx] = input[inIdx];
		}
	`;
}
