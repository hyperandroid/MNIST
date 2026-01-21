import {Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Sum all elements of a tensor to a scalar.
 *
 * Input:  [M, N] (any 2D tensor)
 * Output: [1, 1] containing sum of all elements
 *
 * Used for loss reduction: mean loss = sum(per_sample_loss) / batch_size
 *
 * Note: This is a simple single-pass implementation suitable for small tensors.
 * For large tensors, a hierarchical reduction would be more efficient.
 */
export class SumAllKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		kr: KernelRegistry,
	) {
		super(device, SumAllKernel.sumAllWGSL, kr);

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
			throw new Error("SumAll: input must be 2D tensor");
		}

		const M = input.shape[0];
		const N = input.shape[1];
		const totalSize = M * N;

		if (totalSize > 65536) {
			console.warn("SumAll: tensor size > 65536, consider using hierarchical reduction");
		}

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[1, 1],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: input.buffer}},
				{binding: 1, resource: {buffer: out.buffer}},
				{binding: 2, resource: {buffer: this.paramsBuf}},
			],
		});

		// Single workgroup does the reduction
		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(1, 1, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);

		return out;
	}

	static sumAllWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> input : array<f32>;
		@group(0) @binding(1) var<storage, read_write> output : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;

		var<workgroup> sharedData : array<f32, 256>;

		@compute @workgroup_size(256, 1, 1)
		fn main(
			@builtin(local_invocation_id) lid : vec3<u32>,
		) {
			let totalSize = params.M * params.N;
			let tid = lid.x;

			// Each thread sums a strided portion of the input
			var sum = 0.0;
			var idx = tid;
			while (idx < totalSize) {
				sum = sum + input[idx];
				idx = idx + 256u;
			}

			sharedData[tid] = sum;
			workgroupBarrier();

			// Parallel reduction in sharedData memory
			if (tid < 128u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 128u]; }
			workgroupBarrier();
			if (tid < 64u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 64u]; }
			workgroupBarrier();
			if (tid < 32u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 32u]; }
			workgroupBarrier();
			if (tid < 16u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 16u]; }
			workgroupBarrier();
			if (tid < 8u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 8u]; }
			workgroupBarrier();
			if (tid < 4u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 4u]; }
			workgroupBarrier();
			if (tid < 2u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 2u]; }
			workgroupBarrier();
			if (tid < 1u) { sharedData[tid] = sharedData[tid] + sharedData[tid + 1u]; }
			workgroupBarrier();

			// Thread 0 writes the result
			if (tid == 0u) {
				output[0] = sharedData[0];
			}
		}
	`;
}
