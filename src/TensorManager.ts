import {Tensor} from "./Tensor";

export type ReadBackBufferInfo = {
	buffer: GPUBuffer,
	size: number
}

function alignTo(n: number, multiple: number) {
	return Math.ceil(n / multiple) * multiple;
}

/**
 * Tensor manager is a class responsible of handling tensors.
 * It backs tensors with GPU buffers for reusability and performance.
 */
export class TensorManager {

	tensors = new Map<string, Tensor>();
	readback: ReadBackBufferInfo | null = null;

	private pendingDestroy: GPUBuffer[] = [];

	constructor(
		readonly device: GPUDevice
	) {
	}

	getTensorBuffer(
		name: string,
		usage: GPUBufferUsageFlags,
		shape: number[],
		initialData: Float32Array | undefined = undefined,
	): Tensor {
		const existing = this.tensors.get(name);

		const sizeBytes = shape.reduce((a, b) => a * b, 1) * 4;

		// this buffer fits in the requested one
		if (existing && existing.sizeInBytes() >= sizeBytes && existing.usage === usage) {
			return existing;
		}

		const newBuf = this.device.createBuffer({
			size: alignTo(sizeBytes, 256),
			usage,
		});

		if (existing) {
			this.pendingDestroy.push(existing.buffer);
		}

		const tensor = new Tensor(newBuf, usage, shape);
		this.tensors.set(name, tensor);

		if (initialData) {
			this.writeBufferF32(newBuf, initialData);
		}

		return tensor;
	}

	writeBufferF32(dstBuffer: GPUBuffer, data: Float32Array, dstOffset = 0) {
		this.device.queue.writeBuffer(
			dstBuffer, dstOffset,
			data.buffer, data.byteOffset, data.byteLength
		);
	}

	async readBuffer(
		srcBuffer: GPUBuffer,
		byteLength: number,
		srcOffset = 0
	) {
		const rb = this.ensureReadback(byteLength);

		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(srcBuffer, srcOffset, rb, 0, byteLength);
		this.device.queue.submit([encoder.finish()]);

		await this.device.queue.onSubmittedWorkDone();

		await rb.mapAsync(GPUMapMode.READ, 0, byteLength);
		const mapped = rb.getMappedRange(0, byteLength);
		const copy = mapped.slice(0);
		rb.unmap();

		return new Float32Array(copy);
	}

	private ensureReadback(sizeBytes: number) {
		sizeBytes = alignTo(sizeBytes, 256);
		if (this.readback !== null && this.readback.size >= sizeBytes) {
			return this.readback.buffer;
		}

		if (this.readback) {
			this.pendingDestroy.push(this.readback.buffer);
			this.readback = null;
		}

		const buffer = this.device.createBuffer({
			size: sizeBytes,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		this.readback = {
			buffer,
			size: sizeBytes
		};

		return buffer;
	}

	async flushDestroyQueue() {
		if (this.pendingDestroy.length === 0) {
			return;
		}

		await this.device.queue.onSubmittedWorkDone();
		for (const b of this.pendingDestroy) {
			try {
				b.destroy();
			} catch(e) {
				console.error("Failed to destroy tensor buffer " + e);
			}
		}
		this.pendingDestroy.length = 0;
	}

	async destroyAll() {
		// Wait for GPU to finish anything using these buffers.
		await this.device.queue.onSubmittedWorkDone();

		for (const {buffer} of this.tensors.values()) {
			try {
				buffer.destroy();
			} catch(e) {
				console.error("Failed to destroy tensor buffer "+e);
			}
		}
		this.tensors.clear();

		if (this.readback) {
			try {
				this.readback.buffer.destroy();
			} catch(e) {
				console.error("Failed to destroy readback buffer "+e);
			}
			this.readback = null;
		}

		for (const b of this.pendingDestroy) {
			try {
				b.destroy();
			} catch(e) {
				console.error("Failed to destroy scheduled to destroy tensor buffer "+e);
			}
		}
		this.pendingDestroy.length = 0;
	}
}
