import {Tensor} from "../tensor/Tensor";

/**
 * Base interface for all layers.
 */
export interface Layer {

	inputTensor: Tensor | undefined;
	name: string;

	forward(input: Tensor, isTraining: boolean): Tensor;

	parameters(): Tensor[];
}
