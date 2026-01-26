import {Layer} from "./Layer";
import {Tensor} from "../tensor/Tensor";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {TensorManager} from "../tensor/TensorManager";

export class ReLU implements Layer {

	inputTensor: Tensor | undefined;

	constructor(
		readonly tm: TensorManager,
		readonly kr: KernelRegistry,
		readonly name: string = "ReLU",
	) {

	}

	forward(input: Tensor, isTraining: boolean): Tensor {
		this.inputTensor = input;

		// Use scoped tensor - kernel overwrites entire buffer, no initialization needed
		const output = this.tm.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			input.shape,
		);

		return this.kr.relu.run(input, output);
	}

	parameters(): Tensor[] {
		return [];
	}

}
