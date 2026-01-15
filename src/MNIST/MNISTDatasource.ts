import {Datasource} from "../model/Datasource";

export interface DataSourceIterator {
	hasNext(): boolean;
	next(): number;

	startBatch(): void;
}

class MNISTDataSourceIterator implements DataSourceIterator {

	constructor(
		private batchSize: number,
		private data: Uint8Array,
		private labels: Uint8Array,
	) {

	}

    hasNext(): boolean {
        throw new Error("Method not implemented.");
    }

    next(): number {
        throw new Error("Method not implemented.");
    }

	startBatch(): void {

	}
}

/**
 * A naive MNIST data source.
 * It loads in memory all training and test data.
 */
export class MNISTDatasource implements Datasource {

	trainData: Uint8Array | null = null;
	trainLabelsData: Uint8Array | null = null;

	testData: Uint8Array | null = null;
	testLabelsData: Uint8Array | null = null;

	testImageSize = 28*28;
	testImagesCount = 10000;

	trainImagesCount = 60000;

	constructor() {
	}

	getTrainIterator(batchSize: number): DataSourceIterator {
		return this.getIterator(batchSize, this.trainData!, this.trainLabelsData!);
	}

	private getIterator(
		batchSize: number,
		data: Uint8Array,
		labels: Uint8Array,
	): DataSourceIterator {
		return new MNISTDataSourceIterator(batchSize, data, labels);
	}

	async load() {
		const trainImagesResponse = await fetch("data/train-images.idx3-ubyte")
		this.trainData = new Uint8Array(await trainImagesResponse.arrayBuffer(), 16);
		const trainLabelsResponse = await fetch("data/train-labels.idx1-ubyte");
		this.trainLabelsData = new Uint8Array(await trainLabelsResponse.arrayBuffer(), 8);

		const testDataResponse = await fetch("data/t10k-images.idx3-ubyte");
		this.testData = new Uint8Array(await testDataResponse.arrayBuffer(), 16);
		const testLabelsResponse = await fetch("data/t10k-labels.idx1-ubyte");
		this.testLabelsData = new Uint8Array(await testLabelsResponse.arrayBuffer(), 8);

		this.testImagesCount = this.testData.byteLength / this.testImageSize;
		this.trainImagesCount = this.trainData.byteLength / this.testImageSize;

		console.log(`Test images count ${this.testImagesCount}`);
		console.log(`Train images count ${this.trainImagesCount}`);
		console.log(this.testImageSize);
	}

	/**
	 * generate a random train dataset of size N.
	 * @param N the size of the dataset.
	 * @returns a Uint32Array of indices of the dataset.
	 */
	getRandomTrainDataset(): Uint32Array {
		return this.getRandomDataset(this.trainData!.length);
	}

	/**
	 * generate a random test dataset of size N.
	 * @param N the size of the dataset.
	 * @returns a Uint32Array of indices of the dataset.
	 */
	getRandomTestDataset(): Uint32Array {
		return this.getRandomDataset(this.testData!.length);
	}

	/**
	 * generate a random dataset of size N.
	 * @param size
	 * @private
	 */
	private getRandomDataset(size: number): Uint32Array {
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

	getRandomTrainImageData(): Uint8Array {
		return this.getRandomImageData(this.trainData!, this.trainImagesCount);
	}

	getRandomTestImageData(): Uint8Array {
		return this.getRandomImageData(this.testData!, this.testImagesCount);
	}

	private getRandomImageData(source: Uint8Array, numImages: number) {
		const index = Math.floor(Math.random() * numImages);
		return source!.subarray(this.testImageSize * index, this.testImageSize * (index + 1));
	}

	showRandomImage(data: Uint8Array) {
		const canvas = document.createElement("canvas");
		canvas.width = 28 * 10;
		canvas.height = 28 * 10;
		document.body.appendChild(canvas);
		const ctx = canvas.getContext("2d")!;
		for (let r = 0; r < 28; r++) {
			for (let c = 0; c < 28; c++) {
				const index = r * 28 + c;
				const value = data[index];
				ctx.fillStyle = `rgba(${value}, ${value}, ${value}, 1)`;
				ctx.fillRect(c * 10, r * 10, 10, 10);
			}
		}
	}

}
