import {GradientFunction} from "../GradientFunction";
import {Tensor} from "../../tensor/Tensor";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";

/**
 * Backward for Dropout: Y = X * mask
 *
 * Given dL/dY, computes:
 *   dL/dX = dL/dY * mask
 *
 * Uses the same mask from forward pass (inverted dropout scaling included).
 */
export class DropoutBackward implements GradientFunction {
	readonly name = "DropoutBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): Tensor[] {
		const [mask] = this.savedTensors;

		// dX = gradOutput * mask (same mask used in forward)
		const gradInput = this.kr.elemMul.run(gradOutput, mask);

		return [gradInput];
	}
}
