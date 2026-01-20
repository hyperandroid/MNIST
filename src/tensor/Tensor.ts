import {GradientFunction} from "../autograd/GradientFunction";
import {TensorManager} from "./TensorManager";

/**
 * A Tensor is a multidimensional array.
 * Right now, it will support 2D arrays.
 *
 * Tensors are backed by a WebGPU buffer for extra performance.
 */
export class Tensor {

	readonly size: number;

	// Autograd fields
	gradient?: Tensor;
	gradFn?: GradientFunction = undefined;
	parents?: Tensor[] = undefined;
	requiresGradient: boolean;

	/**
	 * Direct build from existing data that is contiguous in memory. Shape if not set,
	 * will be [1, initalData.length]
	 *
	 * @param name
	 * @param buffer
	 * @param usage
	 * @param shape optional shape of the tensor data.
	 * @param requiresGradient whether this tensor needs gradient computation
	 */
	constructor(
		readonly name: string,
		readonly buffer: GPUBuffer,
		readonly usage: GPUBufferUsageFlags,
		readonly shape: number[],
		requiresGradient: boolean = false,
	) {
		this.size = shape.reduce((a, b) => a * b, 1);
		this.requiresGradient = requiresGradient;
	}

	sizeInBytes() {
		return this.size * 4;
	}

	backward() {
		if (!this.gradient) {
			throw new Error("Tensor has no gradient");
		}
	}

	zeroGrad(tm: TensorManager) {
		if (this.gradient) {
			tm.zeros(this.gradient);
		}
	}
}
