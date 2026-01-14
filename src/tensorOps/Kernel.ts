export function ceilDiv(a: number, b: number) {
	return Math.floor((a + b - 1) / b);
}

/**
 * This class has the internal state for the Tensor operations.
 * It contains a compute pipeline and a shader module for each operation.
 */
export abstract class Kernel {

	readonly pipeline: GPUComputePipeline;
	readonly module: GPUShaderModule;

	constructor(device: GPUDevice, wgsl: string) {

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
