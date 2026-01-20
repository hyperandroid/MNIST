import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";

/**
 * Adds a 1D bias vector to each row of a 2D matrix.
 *
 * Input:  [M, N]
 * Bias:   [N] (1D) or [1, N] (2D)
 * Output: [M, N] where output[i,j] = input[i,j] + bias[j]
 */
export class BiasAddKernel extends Kernel {

	static readonly BIASADD_OUTPUT = "biasadd_out";
	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
	) {
		super(device, BiasAddKernel.biasAddWGSL);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		input: Tensor,
		bias: Tensor,
		out?: Tensor,
	): Tensor {

		if (input.shape.length !== 2) {
			throw new Error("BiasAdd: input must be 2D tensor");
		}

		// Accept bias as [N] or [1, N]
		const biasN = bias.shape.length === 1
			? bias.shape[0]
			: (bias.shape.length === 2 && bias.shape[0] === 1 ? bias.shape[1] : -1);

		if (biasN === -1) {
			throw new Error("BiasAdd: bias must be [N] or [1, N]");
		}

		if (input.shape[1] !== biasN) {
			throw new Error(`BiasAdd: input columns (${input.shape[1]}) must match bias size (${biasN})`);
		}

		const M = input.shape[0];
		const N = input.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			BiasAddKernel.BIASADD_OUTPUT,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: input.buffer}},
				{binding: 1, resource: {buffer: bias.buffer}},
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

	static biasAddWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> bias : array<f32>;
		@group(0) @binding(2) var<storage, read_write> output : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;

		  if (row < params.M && col < params.N) {
		    let idx : u32 = row * params.N + col;
		    output[idx] = input[idx] + bias[col];
		  }
		}
	`;
}
