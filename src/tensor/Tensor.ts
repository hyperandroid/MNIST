/**
 * A Tensor is a multidimensional array.
 * Right now, it will support 2D arrays.
 *
 * Tensors are backed by a WebGPU buffer for extra performance.
 */
export class Tensor {

	readonly size: number;

	gradient?: Tensor;

	/**
	 * Direct build from existing data that is contiguous in memory. Shape if not set,
	 * will be [1, initalData.length]
	 *
	 * @param buffer
	 * @param usage
	 * @param shape optional shape of the tensor data.
	 * @param requiresGradient
	 */
	constructor(
		readonly buffer: GPUBuffer,
		readonly usage: GPUBufferUsageFlags,
		readonly shape: number[],
		readonly requiresGradient: boolean = false,
	) {
		this.size = shape.reduce((a, b) => a * b, 1);
	}

	sizeInBytes() {
		return this.size * 4;
	}
}
