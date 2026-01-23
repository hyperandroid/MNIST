import {Tensor} from "../tensor/Tensor";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";

export function computeBackwardPass(
	tm: TensorManager,
	kr: KernelRegistry,
	loss: Tensor
): void {

	const visited = new Set<Tensor>();
	const order: Tensor[] = [];

	function sort(t: Tensor): void {
		if (visited.has(t) || !t.requiresGradient) {
			return;
		}

		visited.add(t);

		for (const parent of t.parents ?? []) {
			sort(parent);
		}

		order.push(t);
	}

	sort(loss);

	for (const t of order) {
		t.gradient = undefined;
	}

	loss.gradient = tm.scopedOnes(loss.shape);

	// order has topological order of parameters. we need inverse topological order to compute gradients.
	for (let i = order.length - 1; i >= 0; i--) {
		const t = order[i];
		if (!t.gradFn || !t.gradient) {
			continue;
		}

		const inputGrads = t.gradFn.backward(t.gradient);

		for (let j = 0; j < (t.parents ?? []).length; j++) {
			const parent = t.parents![j];
			if (!parent.requiresGradient) {
				continue;
			}

			if (parent.gradient === undefined) {
				// First gradient - initialize the buffer
				parent.gradient =  tm.getTensorBuffer(
						`${parent.name}_grad`,
						GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
						inputGrads[j].shape,
						new Float32Array(inputGrads[j].size).fill(0)
					);
			}

			kr.inplaceAdd.run(parent.gradient, inputGrads[j]);
		}
	}
}
