import {MNISTDatasource, MNISTDataSourceIterator, MNISTDataSourceIteratorValue} from "./MNIST/MNISTDatasource";
import {TensorManager} from "./tensor/TensorManager";
import {KernelRegistry} from "./tensor/kernel/KernelRegistry";
import {MNIST} from "./MNIST/MNIST";

export type TesterState = "idle"
	| "testing"
	| "finished"
	| "cancelling"
	| "cancelled"
	;

export class Tester {
	private state: TesterState = "idle";
	private testCorrect = 0;
	private testTotal = 0;
	private iterator: MNISTDataSourceIterator | null = null;

	constructor(
		readonly tm: TensorManager,
		readonly kernelRegistry: KernelRegistry,
		readonly mnist: MNIST,
		readonly datasource: MNISTDatasource,
		readonly onTestResult: (correct: number, total: number, current: number, size: number,) => void,
		readonly onTestFinished: (correct: number, total: number,) => void,
		readonly batchSize: number = 32,
	) {

	}

	startTesting() {
		if (this.state !== "idle") {
			throw new Error("Cannot start testing while already running.");
		}
		if (!this.iterator) {
			throw new Error("Cannot start testing without initializing. Call initialize() first.");
		}

		this.state = "testing";
		this.testCorrect = 0;
		this.testTotal = 0;

		requestAnimationFrame(this.testStep.bind(this));
	}

	private finished() {
		return this.iterator !== null && !this.iterator.hasNext();
	}

	private async testStep() {

		if (!this.iterator) {
			throw new Error("Cannot test step without initializing tester");
		}

		if (this.state === "cancelling") {
			this.state = "cancelled";
			this.iterator = null;
			return;
		}

		if (this.iterator.hasNext()) {
			await this.testBatch(this.iterator.next());
			this.onTestResult(
				this.testCorrect,
				this.testTotal,
				this.iterator.getCurrentIndex(),
				this.iterator.getSize(),
			)
		} else {
			this.onTestFinished(this.testCorrect, this.testTotal);
		}

		if (!this.finished()) {
			if (this.state === "testing") {
				requestAnimationFrame(this.testStep.bind(this));
			}
		} else {
			this.state = "finished";
			this.onTestFinished(this.testCorrect, this.testTotal);
		}
	}

	async initialize() {

		this.testTotal = 0;
		this.testCorrect = 0;

		await this.datasource
			.load("data/mnist")
			.catch((e: Error) => {
				throw new Error("Failed to load data source: " + e)
			});

		this.iterator = this.datasource.getTestIterator(this.batchSize);
	}

	private async testBatch(data: MNISTDataSourceIteratorValue) {

		const input = this.tm.getTensorBuffer(
			"test_input",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[data.size, 28 * 28],
			data.data,
		);

		// Forward pass (isTraining = false to disable dropout)
		this.tm.beginScope("test");
		const logits = this.mnist.model.forward(input, false);
		const probs = this.kernelRegistry.softmax.run(logits);

		// Read back predictions
		const probsData = await this.tm.readBuffer(probs.buffer, probs.sizeInBytes());

		// Calculate accuracy
		for (let i = 0; i < data.size; i++) {
			// Find predicted class (argmax of probs)
			let maxProb = -Infinity;
			let predicted = 0;
			for (let j = 0; j < 10; j++) {
				const prob = probsData[i * 10 + j];
				if (prob > maxProb) {
					maxProb = prob;
					predicted = j;
				}
			}

			// Find actual class (argmax of one-hot labels)
			let actual = 0;
			for (let j = 0; j < 10; j++) {
				if (data.labels[i * 10 + j] > 0.5) {
					actual = j;
					break;
				}
			}

			if (predicted === actual) {
				this.testCorrect++;
			}
			this.testTotal++;
		}
	}

	cancelTesting() {
		if (this.state === "testing") {
			this.state = "cancelling";
		}
	}
}
