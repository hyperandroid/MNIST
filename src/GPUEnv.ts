
export class GPUEnv {

	static device: GPUDevice;

	/**
	 * initialize the model and get the webgpu device.
	 */
	static async init() {
		if (!navigator.gpu) {
			throw new Error("WebGPU not available in this browser/context.");
		}

		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			throw new Error("Failed to get GPU adapter.");
		}

		GPUEnv.device = await adapter.requestDevice();
	}
}
