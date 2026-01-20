import {GradientFunction} from "../GradientFunction";
import {Tensor} from "../../tensor/Tensor";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";

/**
 * Backward for MatMul: C = A @ B
 *
 * Given dL/dC, computes:
 *   dL/dA = dL/dC @ B^T
 *   dL/dB = A^T @ dL/dC
 */
export class MatMulBackward implements GradientFunction {
	readonly name = "MatMulBackward";

	constructor(
		readonly savedTensors: Tensor[],
		readonly kr: KernelRegistry,
	) {}

	backward(gradOutput: Tensor): Tensor[] {
		const [A, B] = this.savedTensors;

		// dA = gradOutput @ B^T  (if A requires grad)
		const BT = this.kr.transpose.run(B);
		const gradA = A.requiresGradient
			? this.kr.matmul.run(gradOutput, BT)
			: gradOutput; // placeholder, won't be used

		// dB = A^T @ gradOutput  (if B requires grad)
		const AT = this.kr.transpose.run(A);
		const gradB = B.requiresGradient
			? this.kr.matmul.run(AT, gradOutput)
			: gradOutput; // placeholder, won't be used

		return [gradA, gradB];
	}
}
