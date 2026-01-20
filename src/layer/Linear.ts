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

	private matmulout?: Tensor;
	private sumout?: Tensor;

	constructor(
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
		init: LinearInitializer,
	) {
		const shape = [init.inputFeatures, init.outputFeatures];

		this.name = init.name;

		this.weights = tm.getTensorBuffer(
			`${init.name}_w`,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			shape,
			init.initializer(shape, init.inputFeatures),
		);

		if (init.useBias) {
			this.bias = tm.getTensorBuffer(
				`${init.name}_b`,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
				[init.outputFeatures],
			);
		}
	}

	backward(input: Tensor) {
		// TODO
	}

	forward(input: Tensor, isTraining: boolean): Tensor {

		if (this.matmulout === undefined) {
			this.matmulout = this.tm.getTensorBuffer(
				`${this.name}_mmout`,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
				[input.shape[0], this.weights.shape[1]],
			)
		}

		const mm = this.kr.matmul.run(input, this.weights, this.matmulout);
		if (!this.bias) {
			return mm;
		}

		if (this.sumout === undefined) {
			this.sumout = this.tm.getTensorBuffer(
				`${this.name}_sumout`,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
				[input.shape[0], this.weights.shape[1]],
			)
		}

		return this.kr.biasadd.run(this.matmulout, this.bias, this.sumout);
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
