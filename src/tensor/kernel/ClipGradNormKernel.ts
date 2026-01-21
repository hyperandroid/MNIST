import {Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Clip gradient by L2 norm (in-place).
 *
 * If ||gradient|| > maxNorm, scales gradient so ||gradient|| = maxNorm.
 * Otherwise leaves gradient unchanged.
 *
 * Input:  [M, N] gradient tensor (modified in-place)
 * maxNorm: f32 threshold
 *
 * Algorithm:
 * 1. Compute sum of squares using parallel reduction
 * 2. If sqrt(sumSq) > maxNorm, scale all elements by (maxNorm / sqrt(sumSq))
 */
export class ClipGradNormKernel extends Kernel {

	private readonly params = new Float32Array(4); // M, N as u32, maxNorm as f32, padding
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		kr: KernelRegistry,
	) {
		super(device, ClipGradNormKernel.clipGradNormWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		gradient: Tensor,
		maxNorm: number,
	): void {
		if (gradient.shape.length !== 2) {
			throw new Error("ClipGradNorm: gradient must be 2D tensor");
		}

		const M = gradient.shape[0];
		const N = gradient.shape[1];
		const totalSize = M * N;

		if (totalSize > 65536) {
			console.warn("ClipGradNorm: tensor size > 65536, results may be inaccurate");
		}

		// Pack params: M, N as uint32 view, then maxNorm as float32
		const uintView = new Uint32Array(this.params.buffer);
		uintView[0] = M;
		uintView[1] = N;
		this.params[2] = maxNorm;

		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: gradient.buffer}},
				{binding: 1, resource: {buffer: this.paramsBuf}},
			],
		});

		// Single workgroup does the reduction and conditional scaling
		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(1, 1, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);
	}

	static clipGradNormWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		  maxNorm : f32,
		  _pad : u32,
		};

		@group(0) @binding(0) var<storage, read_write> gradient : array<f32>;
		@group(0) @binding(1) var<uniform> params : Params;

		var<workgroup> sharedData : array<f32, 256>;

		@compute @workgroup_size(256, 1, 1)
		fn main(
			@builtin(local_invocation_id) lid : vec3<u32>,
		) {
			let totalSize = params.M * params.N;
			let tid = lid.x;

			// Phase 1: Each thread computes sum of squares for its strided portion
			var sumSq = 0.0;
			var idx = tid;
			while (idx < totalSize) {
				let val = gradient[idx];
				sumSq = sumSq + val * val;
				idx = idx + 256u;
			}

			sharedData[tid] = sumSq;
			workgroupBarrier();

			// Parallel reduction to compute total sum of squares
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

			// Thread 0 computes the scale factor
			let norm = sqrt(sharedData[0]);
			var scale = 1.0;
			if (norm > params.maxNorm) {
				scale = params.maxNorm / norm;
			}

			// Store scale in shared memory for other threads
			if (tid == 0u) {
				sharedData[0] = scale;
			}
			workgroupBarrier();

			let finalScale = sharedData[0];

			// Phase 2: Scale all elements if needed (only if scale != 1.0)
			if (finalScale < 1.0) {
				idx = tid;
				while (idx < totalSize) {
					gradient[idx] = gradient[idx] * finalScale;
					idx = idx + 256u;
				}
			}
		}
	`;
}
