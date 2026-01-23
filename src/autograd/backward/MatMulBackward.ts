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

	backward(gradOutput: Tensor): (Tensor | null)[] {
		const [A, B] = this.savedTensors;

		// dA = gradOutput @ B^T  (only compute if A requires grad)
		let gradA: Tensor | null = null;
		if (A.requiresGradient) {
			const BT = this.kr.transpose.run(B);
			gradA = this.kr.matmul.run(gradOutput, BT);
		}

		// dB = A^T @ gradOutput  (only compute if B requires grad)
		let gradB: Tensor | null = null;
		if (B.requiresGradient) {
			const AT = this.kr.transpose.run(A);
			gradB = this.kr.matmul.run(AT, gradOutput);
		}

		return [gradA, gradB];
	}
}
