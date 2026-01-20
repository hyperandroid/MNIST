import {Tensor} from "../tensor/Tensor";

/**
 * Base interface for all layers.
 */
export interface Layer {
	forward(input: Tensor, isTraining: boolean): Tensor;

	parameters(): Tensor[];
}
