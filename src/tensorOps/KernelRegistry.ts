import {MatMulKernel} from "./MatMulKernel";
import {RELUKernel} from "./RELUKernel";
import {SoftmaxKernel} from "./SoftmaxKernel";
import {TensorManager} from "../TensorManager";

export class KernelRegistry {

	readonly matmul: MatMulKernel;
	readonly relu: RELUKernel;
	readonly softmax: SoftmaxKernel;

	constructor(device: GPUDevice, tm: TensorManager) {
		this.matmul = new MatMulKernel(device, tm);
		this.relu = new RELUKernel(device, tm);
		this.softmax = new SoftmaxKernel(device, tm);
	}
}
