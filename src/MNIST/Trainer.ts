import {MNISTDatasource, MNISTDataSourceIterator, MNISTDataSourceIteratorValue} from "./MNISTDatasource";
import {SGD} from "../optimizer/SGD";
import {Optimizer} from "../optimizer/Optimizer";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {GPUEnv} from "../GPUEnv";
import {computeBackwardPass} from "../autograd/BackwardPass";
import {MNIST} from "./MNIST";

export type TrainerState = "idle"
	| "training"
	| "finished"
	| "cancelling"
	| "cancelled"
	;

export class Trainer {

	private optimizer: Optimizer;
	private iterator: MNISTDataSourceIterator | null = null;

	private currentEpoch = 0;
	private state: TrainerState = "idle";

	saveSnapshot = false;

	constructor(
		readonly tm: TensorManager,
		readonly kernelRegistry: KernelRegistry,
		readonly mnist: MNIST,
		readonly datasource: MNISTDatasource,
		readonly onTrainingFinished: () => void,
		readonly onUpdateData: (epoch: number, epochs: number, current: number, total: number) => void,
		readonly epochs: number = 15,
		readonly batchSize: number = 32,
	) {
		this.optimizer = new SGD(
			mnist.model.parameters(),
			0.05,
			tm,
			kernelRegistry,
			batchSize
		);
	}

	async initialize() {

		this.currentEpoch = 0;

		const trainSize = Math.min(
			this.datasource.trainImagesCount,
			this.datasource.maxTrainSize
		);

		const stepsPerEpoch = Math.ceil(trainSize / this.batchSize);

		this.optimizer.setSchedule({
			type: "cosine",
			minLr: 0.001,
			maxSteps: stepsPerEpoch * this.epochs
		});

		this.iterator = this.datasource.getTrainIterator(this.batchSize);

		// restart model.
		await this.mnist.restart();
	}

	private async snapshot() {
		if (!this.saveSnapshot) {
			return;
		}

		for (const parameter of this.mnist.model.parameters()) {
			const buffer = await this.tm.readBuffer(parameter.buffer, parameter.sizeInBytes());
			const blob = new Blob([buffer], {type: "application/octet-stream"});
			const url = URL.createObjectURL(blob);
			const a = document.createElement("a");
			a.href = url;
			a.download = `model-${this.currentEpoch}-${parameter.name}.bin`;
			a.click();
			URL.revokeObjectURL(url);
		}
	}

	private finished() {
		return this.currentEpoch >= this.epochs;
	}

	private async sync() {
		await GPUEnv.device.queue.onSubmittedWorkDone();
	}

	private async trainBatch(data: MNISTDataSourceIteratorValue) {

		// 1. Zero gradients
		this.optimizer.zeroGrad();
		await GPUEnv.device.queue.onSubmittedWorkDone();

		// 2. Prepare data
		const currentBatchSize = data.size;

		const input = this.tm.getTensorBuffer(
			"input",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[currentBatchSize, 28 * 28],
			data.data,
		);

		const labelsOneHot = this.tm.getTensorBuffer(
			"labels_onehot",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[currentBatchSize, 10],
			data.labels
		);

		// 3. Forward (begin scope for transient tensors)
		this.tm.beginScope("fwd");
		const logits = this.mnist.model.forward(input, true);
		const loss = this.kernelRegistry.crossEntropy.run(logits, labelsOneHot);

		// 4. Backward (scope set inside topologicalSort)
		computeBackwardPass(this.tm, this.kernelRegistry, loss);

		await this.sync();

		// 5. Optimize, SGD
		this.optimizer.step(currentBatchSize);

		await this.sync();
	}

	async trainStep() {

		if (!this.iterator) {
			throw new Error("Cannot train step without initializing trainer");
		}

		if (this.state === "cancelling") {
			this.state = "cancelled";
			this.iterator = null;
			return;
		}

		if (this.iterator.hasNext()) {
			await this.trainBatch(this.iterator.next());
		} else {
			this.iterator = this.datasource.getTrainIterator(this.batchSize);
			this.currentEpoch++;
			await this.snapshot();
		}

		if (!this.finished()) {
			this.onUpdateData(
				this.currentEpoch,
				this.epochs,
				this.iterator.getCurrentIndex(),
				this.iterator.getSize()
			);

			if (this.state === "training") {
				requestAnimationFrame(this.trainStep.bind(this));
			}
		} else {
			// Note: snapshot() already called at line 150 after final epoch increment
			this.state = "finished";
			this.onUpdateData(
				this.currentEpoch,
				this.epochs,
				this.iterator.getCurrentIndex(),
				this.iterator.getSize()
			);
			this.onTrainingFinished();
		}
	}

	startTraining() {
		this.state = "training";
		requestAnimationFrame(() => this.trainStep());
	}

	cancelTraining() {
		if (this.state === "training") {
			this.state = "cancelling";
		}
	}
}
