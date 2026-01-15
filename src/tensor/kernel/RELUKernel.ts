import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";

export class RELUKernel extends Kernel {

	static readonly RELU_OUTPUT = "relu_out";
	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
	) {
		super(device, RELUKernel.reluWGSL);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		t0: Tensor,
		out?: Tensor,
	): Tensor {

		if (
			t0.shape.length !== 2
			|| (out !== undefined && out.shape.length !== 2)
		) {
			throw new Error("RELU: expected 2D tensor");
		}

		const M = t0.shape[0];
		const N = t0.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			RELUKernel.RELU_OUTPUT,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M,N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: t0.buffer}},
				{binding: 1, resource: {buffer: out.buffer}},
				{binding: 2, resource: {buffer: this.paramsBuf}},
			],
		});

		// Dispatch
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

	static reluWGSL = `
    	// RELU: remove negative values from a tensor.

		struct Params {
		  M : u32,
		  N : u32,
		};
		
		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read_write> B : array<f32>;
		@group(0) @binding(2) var<uniform> params : Params;
		
		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;
		
		  let inBounds : bool = (row < params.M) && (col < params.N);

		  // Only write valid output elements
		  if (inBounds) {
		  	let f: f32 = A[row * params.N + col];
			B[row * params.N + col] = max(0f, f);
		  }
		}
	`;
}
