import type {KernelRegistry} from "./KernelRegistry";

export function ceilDiv(a: number, b: number) {
	return Math.floor((a + b - 1) / b);
}

/**
 * This class has the internal state for the Tensor operations.
 * It contains a compute pipeline and a shader module for each operation.
 *
 * For autograd support, kernels can access the full registry via `kr`
 * after it's been set by KernelRegistry.initAutograd().
 */
export abstract class Kernel {

	readonly pipeline: GPUComputePipeline;
	readonly module: GPUShaderModule;

	constructor(
		device: GPUDevice,
		wgsl: string,
		readonly kr: KernelRegistry
	) {

		this.module = device.createShaderModule({code: wgsl});
		this.pipeline = device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.module,
				entryPoint: "main"
			},
		});
	}
}
