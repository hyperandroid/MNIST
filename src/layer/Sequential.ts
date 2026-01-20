import {Layer} from "./Layer";
import {Tensor} from "../tensor/Tensor";

export class Sequential implements Layer {

	readonly layers: Layer[] = [];

	constructor(...seq: Layer[]) {
		for (const l of seq) {
			this.layers.push(l);
		}
	}

	backward(input: Tensor): void {
	}

	forward(input: Tensor, isTraining: boolean): Tensor {
		for (const l of this.layers) {
			input = l.forward(input, isTraining);
		}

		return input;
	}

	parameters(): Tensor[] {
		return this.layers.flatMap(l => l.parameters());
	}
}
