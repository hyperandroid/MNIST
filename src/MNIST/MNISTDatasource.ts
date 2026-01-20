import {Datasource} from "../model/Datasource";


export interface MNISTDataSourceIteratorValue {
	data: Float32Array;
	labels: Float32Array;
	size: number;
}

export class MNISTDataSourceIterator {

	private currentIndex = 0;
	private iteratorSize = 0;
	private datasetIndices: Uint32Array = new Uint32Array(0);

	private readonly workingImageBuffer: Float32Array;
	private readonly workingLabelBuffer: Float32Array;

	constructor(
		readonly batchSize: number,
		private imageData: Float32Array,
		private oneImageSize: number,
		private labelsData: Float32Array,
		private oneLabelSize: number,
	) {
		this.workingImageBuffer = new Float32Array(oneImageSize * this.batchSize);
		this.workingLabelBuffer = new Float32Array(oneLabelSize * this.batchSize);

		const trainImageElemets = this.imageData.length / oneImageSize;
		const trainLabelElemets = this.labelsData.length / oneLabelSize;

		if (trainImageElemets !== trainLabelElemets) {
			throw new Error("MNIST data source: image and label data have different sizes");
		}

		this.iteratorSize = trainLabelElemets;

		this.restart();
	}

    hasNext(): boolean {
		return this.currentIndex < this.iteratorSize;
	}

	/**
	 * populate working buffers with the next batch of data.
	 * @returns the next batch of data.
	 */
    next(): MNISTDataSourceIteratorValue {

		const bs = Math.min(this.batchSize, this.iteratorSize - this.currentIndex);

		for (let i = 0; i < bs; i++) {
			const index = this.datasetIndices[this.currentIndex++];
			const imageIndex = index * this.oneImageSize;
			const labelIndex = index * this.oneLabelSize;

			this.workingImageBuffer.set(
				this.imageData.subarray(imageIndex, imageIndex + this.oneImageSize),
				i * this.oneImageSize
			);

			this.workingLabelBuffer[i] = this.labelsData[labelIndex];
		}

		return {
			data: bs < this.batchSize ?
				this.workingImageBuffer.subarray(0, bs * this.oneImageSize) :
				this.workingImageBuffer,
			labels: bs < this.batchSize ?
				this.workingLabelBuffer.subarray(0, bs) :
				this.workingLabelBuffer,
			size: bs,
		};
	}

	restart(): void {
		this.currentIndex = 0;
		this.datasetIndices = MNISTDataSourceIterator.getRandomDataset(
			this.imageData.length / this.oneImageSize
		);
	}

	/**
	 * generate a random dataset of size N.
	 * @param size
	 * @private
	 */
	static getRandomDataset(size: number): Uint32Array {
		const indices = new Uint32Array(size);
		for(let i = 0; i < indices.length; i++) {
			indices[i] = i;
		}

		for (let i = size - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[indices[i], indices[j]] = [indices[j], indices[i]];
		}

		return indices;
	}

}

/**
 * A naive MNIST data source.
 * It loads in memory all training and test data.
 */
export class MNISTDatasource implements Datasource {

	trainData: Float32Array | null = null;
	trainLabelsData: Float32Array | null = null;

	testData: Float32Array | null = null;
	testLabelsData: Float32Array | null = null;

	testImageSize = 28 * 28;
	testImagesCount = 10000;

	trainImagesCount = 60000;

	constructor() {
	}

	private toFloat32(uint: Uint8Array, div = true): Float32Array {
		const ret = new Float32Array(uint.length);
		for (let i = 0; i < uint.length; i++) {
			ret[i] = uint[i] / (div ? 255 : 1);
		}
		return ret;
	}

	private getIterator(
		batchSize: number,
		data: Float32Array,
		labels: Float32Array,
	): MNISTDataSourceIterator {
		return new MNISTDataSourceIterator(
			batchSize,
			data,
			28 * 28,
			labels,
			1
		);
	}

	getTrainIterator(batchSize: number): MNISTDataSourceIterator {
		if (!this.trainData || !this.trainLabelsData) {
			throw new Error("MNISTDatasource: train data not loaded. Call load() first.");
		}
		return this.getIterator(batchSize, this.trainData, this.trainLabelsData);
	}

	getTestIterator(batchSize: number): MNISTDataSourceIterator {
		if (!this.testData || !this.testLabelsData) {
			throw new Error("MNISTDatasource: test data not loaded. Call load() first.");
		}
		return this.getIterator(batchSize, this.testData, this.testLabelsData);
	}

	async load() {
		const trainImagesResponse = await fetch("data/train-images.idx3-ubyte")
		this.trainData = this.toFloat32(new Uint8Array(await trainImagesResponse.arrayBuffer(), 16));
		const trainLabelsResponse = await fetch("data/train-labels.idx1-ubyte");
		this.trainLabelsData = this.toFloat32(new Uint8Array(await trainLabelsResponse.arrayBuffer(), 8), false);

		const testDataResponse = await fetch("data/t10k-images.idx3-ubyte");
		this.testData = this.toFloat32(new Uint8Array(await testDataResponse.arrayBuffer(), 16));
		const testLabelsResponse = await fetch("data/t10k-labels.idx1-ubyte");
		this.testLabelsData = this.toFloat32(new Uint8Array(await testLabelsResponse.arrayBuffer(), 8), false);

		this.testImagesCount = this.testData.length / this.testImageSize;
		this.trainImagesCount = this.trainData.length / this.testImageSize;

		console.log(`Test images count ${this.testImagesCount}`);
		console.log(`Train images count ${this.trainImagesCount}`);
		console.log(this.testImageSize);
	}

	static ShowRandomImage(data: Float32Array) {
		const canvas = document.createElement("canvas");
		canvas.width = 28 * 10;
		canvas.height = 28 * 10;
		document.body.appendChild(canvas);
		const ctx = canvas.getContext("2d");
		if (!ctx) {
			throw new Error("MNISTDatasource: failed to get 2d canvas context");
		}
		for (let r = 0; r < 28; r++) {
			for (let c = 0; c < 28; c++) {
				const index = r * 28 + c;
				const value = Math.floor(data[index] * 255);
				ctx.fillStyle = `rgba(${value}, ${value}, ${value}, 1)`;
				ctx.fillRect(c * 10, r * 10, 10, 10);
			}
		}
	}

}
