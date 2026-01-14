import {MatMulOp} from "./MatMulOp";

export class OpsRegistry {

	readonly matmul: MatMulOp

	constructor(device: GPUDevice) {
		this.matmul = new MatMulOp(device);
	}
}
