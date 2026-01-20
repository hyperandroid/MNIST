import {Optimizer} from "./Optimizer";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {Tensor} from "../tensor/Tensor";

export class SGD implements Optimizer {
	constructor(
		private params: Tensor[],
		private lr: number,
		private tm: TensorManager,
		private kr: KernelRegistry,
	) {}

	step(): void {
		for (const p of this.params) {
			if (!p.gradient) continue;
			// p = p - lr * grad
			const update = this.kr.scalarMul.run(p.gradient, -this.lr);
			this.kr.inplaceAdd.run(p, update);
		}
	}

	zeroGrad(): void {
		for (const p of this.params) {
			if (p.gradient) {
				this.tm.writeBufferF32(
					p.gradient.buffer,
					new Float32Array(p.gradient.size).fill(0)
				);
			}
		}
	}
}
