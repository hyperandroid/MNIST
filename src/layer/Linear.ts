import {Layer} from "./Layer";
import {Tensor} from "../tensor/Tensor";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";

export type LinearInitializer = {
	name: string;
	readonly inputFeatures: number;
	readonly outputFeatures: number;
	readonly useBias: boolean;
	initializer: (shape: number[], inputFeatures: number) => Float32Array;
}

/**
 * Linear Layer: output = x*W + b
 *
 * Weight layout: [inputFeatures, outputFeatures] (row-major)
 * Input layout:  [batch, inputFeatures]
 * Output layout: [batch, outputFeatures]
 *
 * MatMul computes: input[batch, inputFeatures] * weights[inputFeatures, outputFeatures]
 */
export class Linear implements Layer {

	readonly name: string;

	readonly weights: Tensor;
	readonly bias?: Tensor;

	constructor(
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
		init: LinearInitializer,
	) {
		const shape = [init.inputFeatures, init.outputFeatures];

		this.name = init.name;

		this.weights = tm.getTensorBuffer(
			`${init.name}_weights`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			shape,
			init.initializer(shape, init.inputFeatures),
		);
		this.weights.requiresGradient = true;

		if (init.useBias) {
			this.bias = tm.getTensorBuffer(
				`${init.name}_bias`,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
				[1, init.outputFeatures],
			);
			this.bias.requiresGradient = true;
		}
	}

	forward(input: Tensor, isTraining: boolean): Tensor {

		const matmulout = this.tm.getTensorBuffer(
			`${this.name}_mmout`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[input.shape[0], this.weights.shape[1]],
		)

		const mm = this.kr.matmul.run(input, this.weights, matmulout);
		if (!this.bias) {
			return mm;
		}

		const sumout = this.tm.getTensorBuffer(
			`${this.name}_sumout`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[input.shape[0], this.weights.shape[1]],
		)

		return this.kr.biasadd.run(matmulout, this.bias, sumout);
	}

	parameters(): Tensor[] {
		const ret = [
			this.weights
		];

		if (this.bias !== undefined) {
			ret.push(this.bias);
		}

		return ret;
	}
}
