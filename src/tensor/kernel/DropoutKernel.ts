import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";

/**
 * Applies element-wise dropout by multiplying input with a pre-computed mask.
 *
 * Input:  [M, N]
 * Mask:   [M, N] (values are 0 or scale, where scale = 1/(1-p))
 * Output: [M, N] where output[i,j] = input[i,j] * mask[i,j]
 */
export class DropoutKernel extends Kernel {

	static readonly DROPOUT_OUTPUT = "dropout_out";
	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
	) {
		super(device, DropoutKernel.dropoutWGSL);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		input: Tensor,
		mask: Tensor,
		out?: Tensor,
	): Tensor {

		if (input.shape.length !== 2) {
			throw new Error("Dropout: input must be 2D tensor");
		}

		if (mask.shape.length !== 2) {
			throw new Error("Dropout: mask must be 2D tensor");
		}

		if (input.shape[0] !== mask.shape[0] || input.shape[1] !== mask.shape[1]) {
			throw new Error("Dropout: input and mask shapes must match");
		}

		const M = input.shape[0];
		const N = input.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			DropoutKernel.DROPOUT_OUTPUT,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: input.buffer}},
				{binding: 1, resource: {buffer: mask.buffer}},
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

	static dropoutWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read> mask : array<f32>;
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
		    output[idx] = input[idx] * mask[idx];
		  }
		}
	`;
}
