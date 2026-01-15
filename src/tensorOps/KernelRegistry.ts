import {MatMulKernel} from "./MatMulKernel";
import {RELUKernel} from "./RELUKernel";
import {SoftmaxKernel} from "./SoftmaxKernel";
import {CrossEntropyKernel} from "./CrossEntropyKernel";
import {TensorManager} from "../TensorManager";

export class KernelRegistry {

	readonly matmul: MatMulKernel;
	readonly relu: RELUKernel;
	readonly softmax: SoftmaxKernel;
	readonly crossEntropy: CrossEntropyKernel;

	constructor(device: GPUDevice, tm: TensorManager) {
		this.matmul = new MatMulKernel(device, tm);
		this.relu = new RELUKernel(device, tm);
		this.softmax = new SoftmaxKernel(device, tm);
		this.crossEntropy = new CrossEntropyKernel(device, tm);
	}
}
