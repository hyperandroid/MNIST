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
 * savedTensors: [softmaxOutput, labels]
 */
export class SoftmaxCrossEntropyBackward implements GradientFunction {
	readonly name = "SoftmaxCrossEntropyBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): Tensor[] {
		const [softmaxOutput, labels] = this.savedTensors;

		// dLogits = softmax - labels
		// We need to negate labels and add to softmax
		// Using matadd with negated labels: softmax + (-labels)
		// But we don't have a negate kernel, so we'll use matadd
		// and handle the sign in a custom way.

		// For now, create gradient as softmax - labels using the
		// SubtractKernel if available, or compute inline.
		// Since we only have matadd, we'll need to compute -labels first.

		// Actually, let's use the softmaxCrossEntropyBackward kernel directly
		// which computes softmax - labels in one pass.
		const gradLogits = this.kr.softmaxCEBackward.run(softmaxOutput, labels);

		return [gradLogits];
	}
}
