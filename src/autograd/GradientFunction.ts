import {Tensor} from "../tensor/Tensor";

export interface GradientFunction {

	name: string;
	savedTensors: Tensor[];

	/**
	 * Compute gradients with respect to inputs.
	 * Returns an array matching the number of parents.
	 * null entries indicate gradients that weren't computed (parent doesn't require gradient).
	 */
	backward(gradOutput: Tensor): (Tensor | null)[];
}
