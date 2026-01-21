import {ceilDiv, Kernel} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {MatMulBackward} from "../../autograd/backward/MatMulBackward";
import {KernelRegistry} from "./KernelRegistry";

export class MatMulKernel extends Kernel {

	private readonly params = new Uint32Array(4);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, MatMulKernel.matmulWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		t0: Tensor,
		t1: Tensor,
		out?: Tensor
	): Tensor {
		if (
			t0.shape.length !== 2
			|| t1.shape.length !== 2
			|| (out !== undefined && out.shape.length !== 2)
		) {
			throw new Error("MatMul: expected 2D tensors");
		}

		if (t0.shape[1] !== t1.shape[0]) {
			throw new Error("MatMul: invalid dimensions");
		}

		const M = t0.shape[0];
		const K = t0.shape[1];
		const N = t1.shape[1];

		out = out ?? this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N]);

		if (out.shape[0] !== M || out.shape[1] !== N) {
			throw new Error("MatMul: invalid output dimensions");
		}

		this.params[0] = M;
		this.params[1] = N;
		this.params[2] = K;

		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: t0.buffer}},
				{binding: 1, resource: {buffer: t1.buffer}},
				{binding: 2, resource: {buffer: out.buffer}},
				{binding: 3, resource: {buffer: this.paramsBuf}},
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

		// Autograd: track computation graph
		if (t0.requiresGradient || t1.requiresGradient) {
			out.requiresGradient = true;
			out.parents = [t0, t1];
			out.gradFn = new MatMulBackward([t0, t1], this.kr!);
		}

		return out;
	}

	static matmulWGSL = `
    	// C[M,N] = A[M,K] * B[K,N]  (row-major flat buffers)

		struct Params {
		  M : u32,
		  N : u32,
		  K : u32,
		  _pad : u32,
		};
		
		@group(0) @binding(0) var<storage, read> A : array<f32>;
		@group(0) @binding(1) var<storage, read> B : array<f32>;
		@group(0) @binding(2) var<storage, read_write> C : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;
		
		const TILE : u32 = 16u;
		
		var<workgroup> tileA : array<f32, 16u * 16u>;
		var<workgroup> tileB : array<f32, 16u * 16u>;
		
		@compute @workgroup_size(16, 16, 1)
		fn main(
		  @builtin(global_invocation_id) gid : vec3<u32>,
		  @builtin(local_invocation_id)  lid : vec3<u32>,
		) {
		  let row : u32 = gid.y;
		  let col : u32 = gid.x;
		
		  let inBounds : bool = (row < params.M) && (col < params.N);
		
		  let lidx : u32 = lid.y * TILE + lid.x;
		
		  var acc : f32 = 0.0;
		  let numTiles : u32 = (params.K + TILE - 1u) / TILE;
		
		  for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
			let kBase : u32 = t * TILE;
		
			// Load A tile element (or 0)
			let aCol : u32 = kBase + lid.x;
			if ((row < params.M) && (aCol < params.K)) {
			  tileA[lidx] = A[row * params.K + aCol];
			} else {
			  tileA[lidx] = 0.0;
			}
		
			// Load B tile element (or 0)
			let bRow : u32 = kBase + lid.y;
			if ((bRow < params.K) && (col < params.N)) {
			  tileB[lidx] = B[bRow * params.N + col];
			} else {
			  tileB[lidx] = 0.0;
			}
		
			workgroupBarrier();
		
			for (var i : u32 = 0u; i < TILE; i = i + 1u) {
			  acc = acc + tileA[lid.y * TILE + i] * tileB[i * TILE + lid.x];
			}
		
			workgroupBarrier();
		  }
		
		  // Only write valid output elements
		  if (inBounds) {
			C[row * params.N + col] = acc;
		  }
		}
	`;
}
