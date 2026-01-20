import {GradientFunction} from "../GradientFunction";
import {Tensor} from "../../tensor/Tensor";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";

/**
 * Backward for MatAdd: C = A + B
 *
 * Given dL/dC, computes:
 *   dL/dA = dL/dC
 *   dL/dB = dL/dC
 *
 * Both gradients are just the upstream gradient (identity).
 */
export class MatAddBackward implements GradientFunction {
	readonly name = "MatAddBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): Tensor[] {
		// Both inputs get the same gradient
		return [gradOutput, gradOutput];
	}
}
