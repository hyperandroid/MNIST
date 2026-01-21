import {Kernel, ceilDiv} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Sum reduction along axis 0 (columns).
 *
 * Input:  [M, N]
 * Output: [1, N] where output[0,j] = sum(input[i,j] for all i)
 *
 * Used for computing bias gradients: db = sum(dOut, axis=0)
 */
export class SumReduceKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, SumReduceKernel.sumReduceWGSL, kr);

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
			throw new Error("SumReduce: input must be 2D tensor");
		}

		const M = input.shape[0];
		const N = input.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[1, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: input.buffer}},
				{binding: 1, resource: {buffer: out.buffer}},
				{binding: 2, resource: {buffer: this.paramsBuf}},
			],
		});

		// One thread per column
		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(ceilDiv(N, 256), 1, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);

		return out;
	}

	static sumReduceWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
			let col = gid.x;
			if (col >= params.N) {
				return;
			}

			var sum = 0.0;
			for (var row = 0u; row < params.M; row = row + 1u) {
				sum = sum + input[row * params.N + col];
			}

			output[col] = sum;
		}
	`;
}
