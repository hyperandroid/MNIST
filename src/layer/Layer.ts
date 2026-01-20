import {Tensor} from "../tensor/Tensor";

/**
 * Base interface for all layers.
 */
export interface Layer {
	forward(input: Tensor, isTraining: boolean): Tensor;

	/**
	 * TODO: return Tensor
	 * @param input
	 */
	backward(input: Tensor): void;

	parameters(): Tensor[];
}
