import {MatMulKernel} from "./MatMulKernel";
import {MatAddKernel} from "./MatAddKernel";
import {BiasAddKernel} from "./BiasAddKernel";
import {RELUKernel} from "./RELUKernel";
import {SoftmaxKernel} from "./SoftmaxKernel";
import {CrossEntropyKernel} from "./CrossEntropyKernel";
import {DropoutKernel} from "./DropoutKernel";
import {TensorManager} from "../TensorManager";

export class KernelRegistry {

	readonly matmul: MatMulKernel;
	readonly matadd: MatAddKernel;
	readonly biasadd: BiasAddKernel;
	readonly relu: RELUKernel;
	readonly softmax: SoftmaxKernel;
	readonly crossEntropy: CrossEntropyKernel;
	readonly dropout: DropoutKernel;

	constructor(device: GPUDevice, tm: TensorManager) {
		this.matmul = new MatMulKernel(device, tm);
		this.matadd = new MatAddKernel(device, tm);
		this.biasadd = new BiasAddKernel(device, tm);
		this.relu = new RELUKernel(device, tm);
		this.softmax = new SoftmaxKernel(device, tm);
		this.crossEntropy = new CrossEntropyKernel(device, tm);
		this.dropout = new DropoutKernel(device, tm);
	}
}
