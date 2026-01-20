import {Tensor} from "../tensor/Tensor";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";

export function topologicalSort(
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

	loss.gradient = tm.ones(loss.shape, `${loss.name}_gradient`);

	// order has topological order of parameters.
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

			if (parent.gradient !== undefined) {
				parent.gradient = kr.matadd.run(parent.gradient, inputGrads[j], );
			} else {
				parent.gradient = inputGrads[j];
			}
		}
	}
}
