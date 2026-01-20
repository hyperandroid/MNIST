import {GradientFunction} from "../GradientFunction";
import {Tensor} from "../../tensor/Tensor";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";

/**
 * Backward for ReLU: Y = max(0, X)
 *
 * Given dL/dY, computes:
 *   dL/dX = dL/dY * (X > 0)
 *
 * savedTensors[0] is the original input X from forward pass.
 */
export class ReLUBackward implements GradientFunction {
	readonly name = "ReLUBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): Tensor[] {
		// savedTensors[0] is the original input X
		const [savedInput] = this.savedTensors;

		// dX = gradOutput * (savedInput > 0)
		const gradInput = this.kr.reluBackward.run(gradOutput, savedInput);

		return [gradInput];
	}
}
