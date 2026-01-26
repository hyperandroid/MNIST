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

	// Scoped allocation for transient tensors
	private scopeName = "";
	private scopeCounter = 0;

	constructor(
		readonly device: GPUDevice
	) {
	}

	/**
	 * Begin a named scope for transient tensor allocation.
	 * Resets the counter so tensors get reusable names like "_fwd_0", "_fwd_1", etc.
	 */
	beginScope(name: string) {
		this.scopeName = name;
		this.scopeCounter = 0;
	}

	/**
	 * Get a transient tensor within the current scope.
	 * Uses sequential naming (_scope_0, _scope_1, ...) that resets each scope,
	 * enabling buffer reuse across iterations.
	 */
	getScopedTensor(
		usage: GPUBufferUsageFlags,
		shape: number[],
		initialData?: Float32Array,
	): Tensor {
		const name = `_${this.scopeName}_${this.scopeCounter++}`;
		return this.getTensorBuffer(name, usage, shape, initialData);
	}

	getTensorBuffer(
		name: string,
		usage: GPUBufferUsageFlags,
		shape: number[],
		initialData: Float32Array | undefined = undefined,
	): Tensor {
		const existing = this.tensors.get(name);

		const sizeBytes = shape.reduce((a, b) => a * b, 1) * 4;

		// this buffer fits in the requested one - reuse buffer but update shape
		if (existing && existing.sizeInBytes() >= sizeBytes && existing.usage === usage) {
			if (initialData) {
				this.writeBufferF32(existing.buffer, initialData);
			}

			// Create new tensor with updated shape if shape changed
			const shapeMatch = existing.shape.length === shape.length
				&& existing.shape.every((v, i) => v === shape[i]);

			if (shapeMatch) {
				return existing;
			}

			const updated = new Tensor(name, existing.buffer, usage, shape);
			this.tensors.set(name, updated);
			return updated;
		}

		const newBuf = this.device.createBuffer({
			size: alignTo(sizeBytes, 256),
			usage,
		});

		if (existing) {
			this.pendingDestroy.push(existing.buffer);
		}

		const tensor = new Tensor(name, newBuf, usage, shape);
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
		/*
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

		 */
		const rb = this.ensureReadback(byteLength);

		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(srcBuffer, srcOffset, rb, 0, byteLength);
		this.device.queue.submit([encoder.finish()]);

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

	ones(shape: number[], name: string = "ones") {
		const buf = this.getTensorBuffer(
			name,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			shape);
		this.writeBufferF32(buf.buffer, new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1));
		return buf;
	}

	/**
	 * Create a scoped tensor filled with ones.
	 */
	scopedOnes(shape: number[]) {
		const buf = this.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			shape);
		this.writeBufferF32(buf.buffer, new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1));
		return buf;
	}

	/**
	 * Create a scoped tensor filled with zeros.
	 */
	scopedZeros(shape: number[]) {
		const buf = this.getScopedTensor(
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			shape);
		this.writeBufferF32(buf.buffer, new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(0));
		return buf;
	}

	zeros(buf: Tensor) {
		this.writeBufferF32(buf.buffer, new Float32Array(buf.size).fill(0));
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
