/**
 * A Tensor is a multidimensional array.
 * Right now, it will support 2D arrays.
 *
 * Tensors are backed by a WebGPU buffer for extra performance.
 */
export class Tensor {

	readonly size: number;

	/**
	 * Direct build from existing data that is contiguous in memory. Shape it not set,
	 * will be [1, initalData.length]
	 *
	 * @param buffer
	 * @param usage
	 * @param shape optional shape of the tensor data.
	 */
	constructor(
		readonly buffer: GPUBuffer,
		readonly usage: GPUBufferUsageFlags,
		readonly shape: number[]
	) {
		this.size = shape.reduce((a, b) => a * b, 1);
	}

	sizeInBytes() {
		return this.size * 4;
	}
}
