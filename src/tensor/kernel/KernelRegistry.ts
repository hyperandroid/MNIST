import {MatMulKernel} from "./MatMulKernel";
import {MatAddKernel} from "./MatAddKernel";
import {BiasAddKernel} from "./BiasAddKernel";
import {RELUKernel} from "./RELUKernel";
import {SoftmaxKernel} from "./SoftmaxKernel";
import {CrossEntropyKernel} from "./CrossEntropyKernel";
import {DropoutKernel} from "./DropoutKernel";
import {TransposeKernel} from "./TransposeKernel";
import {SumReduceKernel} from "./SumReduceKernel";
import {ElementwiseMulKernel} from "./ElementwiseMulKernel";
import {ReLUBackwardKernel} from "./ReLUBackwardKernel";
import {SoftmaxCEBackwardKernel} from "./SoftmaxCEBackwardKernel";
import {SoftmaxBackwardKernel} from "./SoftmaxBackwardKernel";
import {ScalarMulKernel} from "./ScalarMulKernel";
import {InplaceAddKernel} from "./InplaceAddKernel";
import {SumAllKernel} from "./SumAllKernel";
import {TensorManager} from "../TensorManager";

export class KernelRegistry {

	readonly matmul: MatMulKernel;
	readonly matadd: MatAddKernel;
	readonly biasadd: BiasAddKernel;
	readonly relu: RELUKernel;
	readonly softmax: SoftmaxKernel;
	readonly crossEntropy: CrossEntropyKernel;
	readonly dropout: DropoutKernel;

	// Autograd support kernels
	readonly transpose: TransposeKernel;
	readonly sumReduce: SumReduceKernel;
	readonly elemMul: ElementwiseMulKernel;
	readonly reluBackward: ReLUBackwardKernel;
	readonly softmaxBackward: SoftmaxBackwardKernel;
	readonly softmaxCEBackward: SoftmaxCEBackwardKernel;

	// Optimizer support kernels
	readonly scalarMul: ScalarMulKernel;
	readonly inplaceAdd: InplaceAddKernel;
	readonly sumAll: SumAllKernel;

	constructor(device: GPUDevice, tm: TensorManager) {
		this.matmul = new MatMulKernel(device, tm, this);
		this.matadd = new MatAddKernel(device, tm, this);
		this.biasadd = new BiasAddKernel(device, tm, this);
		this.relu = new RELUKernel(device, tm, this);
		this.softmax = new SoftmaxKernel(device, tm, this);
		this.crossEntropy = new CrossEntropyKernel(device, tm, this);
		this.dropout = new DropoutKernel(device, tm, this);

		// Autograd support kernels
		this.transpose = new TransposeKernel(device, tm, this);
		this.sumReduce = new SumReduceKernel(device, tm, this);
		this.elemMul = new ElementwiseMulKernel(device, tm, this);
		this.reluBackward = new ReLUBackwardKernel(device, tm, this);
		this.softmaxBackward = new SoftmaxBackwardKernel(device, tm, this);
		this.softmaxCEBackward = new SoftmaxCEBackwardKernel(device, tm, this);

		// Optimizer support kernels
		this.scalarMul = new ScalarMulKernel(device, tm, this);
		this.inplaceAdd = new InplaceAddKernel(device, tm, this);
		this.sumAll = new SumAllKernel(device, tm, this);
	}

}
