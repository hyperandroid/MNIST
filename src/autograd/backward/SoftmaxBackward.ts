import {GradientFunction} from "../GradientFunction";
import {Tensor} from "../../tensor/Tensor";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";

/**
 * Softmax backward.
 *
 * dL/dX = S * (dS - sum(dS * S, axis=1, keepdims=True))
 *
 * where S = softmax(X) and dS = dL/dSoftmax
 */
export class SoftmaxBackward implements GradientFunction {
	readonly name = "SoftmaxBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): Tensor[] {
		// savedTensors: [input, softmaxOutput]
		const softmaxOut = this.savedTensors[1];

		// dX = S * (dS - dot)  where dot = sum(dS * S, per row)
		// This is computed by the softmaxBackward kernel
		const gradInput = this.kr.softmaxBackward.run(gradOutput, softmaxOut);

		return [gradInput];
	}
}
