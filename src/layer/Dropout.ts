import {Layer} from "./Layer";
import {Tensor} from "../tensor/Tensor";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";

/**
 * Dropout layer for regularization.
 *
 * During training: randomly zeroes elements with probability p,
 * scales remaining elements by 1/(1-p) to maintain expected values.
 *
 * During inference: passes input through unchanged.
 *
 * Input/Output layout: [batch, features]
 */
export class Dropout implements Layer {

	readonly name: string;
	readonly p: number;
	readonly scale: number;

	private maskData?: Float32Array;

	constructor(
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
		name: string,
		p: number = 0.5,
	) {
		if (p < 0 || p >= 1) {
			throw new Error("Dropout: p must be in [0, 1)");
		}

		this.name = name;
		this.p = p;
		this.scale = 1 / (1 - p);
	}

	forward(input: Tensor, isTraining: boolean): Tensor {
		if (!isTraining) {
			return input;
		}

		const M = input.shape[0];
		const N = input.shape[1];
		const size = M * N;

		// Ensure mask buffer exists and is correct size
		if (this.maskData === undefined || this.maskData.length < size) {
			this.maskData = new Float32Array(size);
		}

		// Generate random mask on CPU: 0 with prob p, scale with prob (1-p)
		for (let i = 0; i < size; i++) {
			this.maskData[i] = Math.random() < this.p ? 0 : this.scale;
		}

		// Upload mask to GPU (scoped tensor - only needed for this batch)
		const mask = this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			[M, N],
			this.maskData,
		);

		// Output buffer (scoped tensor - only needed for this batch)
		const out = this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[M, N],
		);

		return this.kr.dropout.run(input, mask, out);
	}

	parameters(): Tensor[] {
		return []; // Dropout has no learnable parameters
	}

}
