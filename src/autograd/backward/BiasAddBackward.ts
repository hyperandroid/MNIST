import {GradientFunction} from "../GradientFunction";
import {Tensor} from "../../tensor/Tensor";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";

/**
 * Backward for BiasAdd: Y = X + b (broadcast)
 *
 * Given dL/dY [M, N], computes:
 *   dL/dX = dL/dY                    (same shape)
 *   dL/db = sum(dL/dY, axis=0)       (reduce to [1, N] or [N])
 */
export class BiasAddBackward implements GradientFunction {
	readonly name = "BiasAddBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): (Tensor | null)[] {
		const [input, bias] = this.savedTensors;

		// dX = gradOutput (identity, same shape)
		const gradInput = input.requiresGradient ? gradOutput : null;

		// db = sum(gradOutput, axis=0) -> [1, N] (only compute if bias requires grad)
		const gradBias = bias.requiresGradient
			? this.kr.sumReduce.run(gradOutput)
			: null;

		return [gradInput, gradBias];
	}
}
