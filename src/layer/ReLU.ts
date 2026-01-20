import {Layer} from "./Layer";
import {Tensor} from "../tensor/Tensor";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {TensorManager} from "../tensor/TensorManager";

export class ReLU implements Layer {

	private output?: Tensor;

	constructor(
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
		readonly name: string = "ReLU",
	) {

	}

	backward(input: Tensor): void {
	}

	forward(input: Tensor, isTraining: boolean): Tensor {
		if (this.output === undefined) {
			this.output = this.tm.getTensorBuffer(
				`${this.name}_out`,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
				input.shape,
				new Float32Array(input.size)
			);
		}

		return this.kr.relu.run(input, this.output);
	}

	parameters(): Tensor[] {
		return [];
	}

}
