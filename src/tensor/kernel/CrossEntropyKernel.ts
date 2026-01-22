import {Kernel, ceilDiv} from "./Kernel";
import {Tensor} from "../Tensor";
import {TensorManager} from "../TensorManager";
import {SoftmaxCrossEntropyBackward} from "../../autograd/backward/SoftmaxCrossEntropyBackward";
import {KernelRegistry} from "./KernelRegistry";

/**
 * Cross Entropy Loss kernel (with built-in log-softmax).
 * Computes: -sum(labels * log_softmax(logits)) per row.
 *
 * Input logits: raw logits [M, N] (NOT softmax output).
 * Input labels: one-hot encoded [M, N].
 * Output: per-sample loss [M, 1].
 *
 * Uses logsumexp for numerical stability.
 */
export class CrossEntropyKernel extends Kernel {

	private readonly params = new Uint32Array(2);
	private readonly paramsBuf: GPUBuffer;

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
	) {
		super(device, CrossEntropyKernel.xentropyWGSL, kr);

		this.paramsBuf = device.createBuffer({
			size: 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
	}

	run(
		logits: Tensor,
		labels: Tensor,
		out?: Tensor,
	): Tensor {
		if (logits.shape.length !== 2) {
			throw new Error("CrossEntropy: logits must be 2D tensor");
		}
		if (labels.shape.length !== 2) {
			throw new Error("CrossEntropy: labels must be 2D tensor");
		}
		if (logits.shape[0] !== labels.shape[0] || logits.shape[1] !== labels.shape[1]) {
			throw new Error("CrossEntropy: logits and labels must have same shape");
		}

		const M = logits.shape[0];
		const N = logits.shape[1];

		this.params[0] = M;
		this.params[1] = N;
		this.device.queue.writeBuffer(this.paramsBuf, 0, this.params);

		out = out ?? this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, 1],
		);

		const bindGroup = this.device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: logits.buffer}},
				{binding: 1, resource: {buffer: labels.buffer}},
				{binding: 2, resource: {buffer: out.buffer}},
				{binding: 3, resource: {buffer: this.paramsBuf}},
			],
		});

		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(ceilDiv(M, 256), 1, 1);
		pass.end();

		this.device.queue.submit([encoder.finish()]);

		// Autograd: track computation graph
		// Use combined SoftmaxCrossEntropy backward for numerical stability
		// Forward computes log-softmax internally; backward computes softmax(logits) - labels
		if (logits.requiresGradient) {
			out.requiresGradient = true;
			out.parents = [logits];
			out.gradFn = new SoftmaxCrossEntropyBackward([logits, labels], this.kr!);
		}

		return out;
	}

	static xentropyWGSL = `
		struct Params {
		  M : u32,
		  N : u32,
		};
		
		@group(0) @binding(0) var<storage, read> logits : array<f32>;
		@group(0) @binding(1) var<storage, read> labels : array<f32>; // one-hot
		@group(0) @binding(2) var<storage, read_write> loss : array<f32>;
		@group(0) @binding(3) var<uniform> params : Params;
		
		@compute @workgroup_size(256, 1, 1)
		fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
		  let row = gid.x;
		  if (row >= params.M) { return; }
		
		  let N = params.N;
		  let base = row * N;
		
		  // 1) max logit for stability
		  var m = logits[base];
		  for (var i = 1u; i < N; i = i + 1u) {
			let z = logits[base + i];
			if (z > m) { m = z; }
		  }
		
		  // 2) logsumexp
		  var sumExp = 0.0;
		  for (var i = 0u; i < N; i = i + 1u) {
			sumExp = sumExp + exp(logits[base + i] - m);
		  }
		  let logSumExp = log(sumExp) + m;
		
		  // 3) cross entropy: -sum y_i * (z_i - logsumexp)
		  var ce = 0.0;
		  for (var i = 0u; i < N; i = i + 1u) {
			let y = labels[base + i];
			let z = logits[base + i];
			ce = ce + y * (logSumExp - z);
		  }
		
		  loss[row] = ce;
		}

	`;
}
