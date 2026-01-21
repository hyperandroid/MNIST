import {GradientFunction} from "../GradientFunction";
import {Tensor} from "../../tensor/Tensor";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";

/**
 * Backward for combined Softmax + CrossEntropy.
 *
 * Forward: L = CrossEntropy(Softmax(logits), labels)
 *
 * Given dL/dL = 1 (scalar loss), computes:
 *   dL/dLogits = Softmax(logits) - labels
 *
 * This is numerically more stable than computing them separately.
 * savedTensors: [logits, labels]
 */
export class SoftmaxCrossEntropyBackward implements GradientFunction {
	readonly name = "SoftmaxCrossEntropyBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): Tensor[] {
		const [logits, labels] = this.savedTensors;

		// dLogits = softmax(logits) - labels
		// The kernel computes softmax internally for numerical stability
		const gradLogits = this.kr.softmaxCEBackward.run(logits, labels);

		return [gradLogits];
	}
}
