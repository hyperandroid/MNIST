import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";

export class MatAddKernel extends Kernel {

	static readonly MATADD_OUTPUT = "matadd_out";
	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
	) {
		super(device, MatAddKernel.matAddWGSL);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		t0: Tensor,
		t1: Tensor,
		out?: Tensor,
	): Tensor {

		if (
			t0.shape.length !== 2
			|| t1.shape.length !== 2
			|| (out !== undefined && out.shape.length !== 2)
		) {
			throw new Error("MatAdd: expected 2D tensors");
		}

		if (t0.shape[0] !== t1.shape[0] || t0.shape[1] !== t1.shape[1]) {
			throw new Error("MatAdd: tensor shapes must match");
		}

		const M = t0.shape[0];
		const N = t0.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getTensorBuffer(
			MatAddKernel.MATADD_OUTPUT,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: t0.buffer}},
				{binding: 1, resource: {buffer: t1.buffer}},
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

	static matAddWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};

		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read> B : array<f32>;
		@group(0) @binding(2) var<storage, read_write> C : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;

		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;

		  if (row < params.M && col < params.N) {
		    let idx : u32 = row * params.N + col;
		    C[idx] = A[idx] + B[idx];
		  }
		}
	`;
}
