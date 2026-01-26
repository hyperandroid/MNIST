import {Layer} from "./Layer";
import {Tensor} from "../tensor/Tensor";
import {TensorManager} from "../tensor/TensorManager";

export class Sequential implements Layer {

	inputTensor: Tensor | undefined;

	readonly layers: Layer[] = [];

	constructor(
		...seq: Layer[]
	) {
		for (const l of seq) {
			this.layers.push(l);
		}
	}

	forward(input: Tensor, isTraining: boolean): Tensor {
		this.inputTensor = input;

		for (const l of this.layers) {
			input = l.forward(input, isTraining);
		}

		return input;
	}

	parameters(): Tensor[] {
		return this.layers.flatMap(l => l.parameters());
	}

	zeroGrad(tm: TensorManager): void {
		for (const param of this.parameters()) {
			if (param.gradient) {
				tm.writeBufferF32(
					param.gradient.buffer,
					new Float32Array(param.size).fill(0)
				);
			}
		}
	}

}
