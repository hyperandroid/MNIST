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
		private maxSize: number,
	) {
		this.workingImageBuffer = new Float32Array(oneImageSize * this.batchSize);
		this.workingLabelBuffer = new Float32Array(oneLabelSize * this.batchSize);

		const trainImageElemets = Math.min(this.maxSize, this.imageData.length / oneImageSize);
		const trainLabelElemets = Math.min(this.maxSize, this.labelsData.length / oneLabelSize);

		if (trainImageElemets !== trainLabelElemets) {
			throw new Error("MNIST data source: image and label data have different sizes");
		}

		this.iteratorSize = trainLabelElemets;

		this.restart();
	}

	getCurrentIndex() {
		return this.currentIndex;
	}

	getSize() {
		return this.iteratorSize;
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
			this.workingImageBuffer.set(
				this.imageData.subarray(imageIndex, imageIndex + this.oneImageSize),
				i * this.oneImageSize
			);

			const labelIndex = index * this.oneLabelSize;
			this.workingLabelBuffer.set(
				this.labelsData.subarray(labelIndex, labelIndex + this.oneLabelSize),
				i * this.oneLabelSize
			);
		}

		return {
			data: bs < this.batchSize ?
				this.workingImageBuffer.subarray(0, bs * this.oneImageSize) :
				this.workingImageBuffer,
			labels: bs < this.batchSize ?
				this.workingLabelBuffer.subarray(0, bs * this.oneLabelSize) :
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

	static readonly imageSize = 28 * 28;
	testImagesCount = 10000;

	trainImagesCount = 60000;

	maxTrainSize = Infinity;
	maxTestSize = Infinity;

	constructor() {
	}

	private toFloat32(uint: Uint8Array): Float32Array {
		const ret = new Float32Array(uint.length);
		for (let i = 0; i < uint.length; i++) {
			ret[i] = uint[i] / 255;
		}
		return ret;
	}

	private getIterator(
		batchSize: number,
		data: Float32Array,
		labels: Float32Array,
		maxSize: number,
	): MNISTDataSourceIterator {
		return new MNISTDataSourceIterator(
			batchSize,
			data,
			MNISTDatasource.imageSize,
			labels,
			10,
			maxSize,
		);
	}

	getTrainIterator(batchSize: number): MNISTDataSourceIterator {
		if (!this.trainData || !this.trainLabelsData) {
			throw new Error("MNISTDatasource: train data not loaded. Call load() first.");
		}
		return this.getIterator(batchSize, this.trainData, this.trainLabelsData, this.maxTrainSize);
	}

	getTestIterator(batchSize: number): MNISTDataSourceIterator {
		if (!this.testData || !this.testLabelsData) {
			throw new Error("MNISTDatasource: test data not loaded. Call load() first.");
		}
		return this.getIterator(batchSize, this.testData, this.testLabelsData, this.maxTestSize);
	}

	private onehot(uint: Uint8Array): Float32Array {
		const output = new Float32Array(uint.length * 10);

		for (let i = 0; i < uint.length; i++) {
			const index = uint[i];
			output[i * 10 + index] = 1.0;
		}

		return output;
	}

	async load(path: string) {
		const trainImagesResponse = await fetch(`${path}/train-images.idx3-ubyte`)
		this.trainData = this.toFloat32(new Uint8Array(await trainImagesResponse.arrayBuffer(), 16));
		const trainLabelsResponse = await fetch(`${path}/train-labels.idx1-ubyte`);
		this.trainLabelsData = this.onehot(new Uint8Array(await trainLabelsResponse.arrayBuffer(), 8));

		const testDataResponse = await fetch(`${path}/t10k-images.idx3-ubyte`);
		this.testData = this.toFloat32(new Uint8Array(await testDataResponse.arrayBuffer(), 16));
		const testLabelsResponse = await fetch(`${path}/t10k-labels.idx1-ubyte`);
		this.testLabelsData = this.onehot(new Uint8Array(await testLabelsResponse.arrayBuffer(), 8));

		this.testImagesCount = this.testData.length / MNISTDatasource.imageSize;
		this.trainImagesCount = this.trainData.length / MNISTDatasource.imageSize;
	}
}
