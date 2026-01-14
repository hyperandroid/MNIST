import {MatMulOp} from "./MatMulOp";
import {RELUOp} from "./RELUOp";
import {TensorManager} from "../TensorManager";

export class OpsRegistry {

	readonly matmul: MatMulOp;
	readonly relu: RELUOp;

	constructor(device: GPUDevice, tm: TensorManager) {
		this.matmul = new MatMulOp(device, tm);
		this.relu = new RELUOp(device, tm);
	}
}
