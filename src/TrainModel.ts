import {GPUEnv} from "./GPUEnv";
import {KernelRegistry} from "./tensor/kernel/KernelRegistry";
import {TensorManager} from "./tensor/TensorManager";
import {MNISTDatasource, MNISTDataSourceIterator} from "./MNIST/MNISTDatasource";
import {Sequential} from "./layer/Sequential";
import {Linear} from "./layer/Linear";
import {heNormal, heUniform} from "./math/Utils";
import {ReLU} from "./layer/ReLU";
import {Dropout} from "./layer/Dropout";
import {SGD} from "./optimizer/SGD";
import {topologicalSort} from "./autograd/TopologicalSort";

/**
 * Initialize the GPU. If not present, no training will be possible.
 */
await GPUEnv.init()

const tm = new TensorManager(GPUEnv.device);
const kernelRegistry = new KernelRegistry(GPUEnv.device, tm);

const model = new Sequential(
	new Linear(tm, kernelRegistry, {
		name: "first",
		inputFeatures: 28*28,
		outputFeatures: 128,
		useBias: true,
		initializer: heUniform
	}),
	new ReLU(tm, kernelRegistry, "ReLU1"),
	new Linear(tm, kernelRegistry, {
		name: "second",
		inputFeatures: 128,
		outputFeatures: 10,
		useBias: true,
		initializer: heUniform
	}),

	/*
	new Linear(tm, kernelRegistry, {
		name: "first",
		inputFeatures: 28*28,
		outputFeatures: 256,
		useBias: true,
		initializer: heNormal
	}),
	new ReLU(tm, kernelRegistry, "ReLU1"),

	new Dropout(tm, kernelRegistry, "dropout1", 0.5),

	new Linear(tm, kernelRegistry, {
		name: "second",
		inputFeatures: 256,
		outputFeatures: 128,
		useBias: true,
		initializer: heNormal
	}),
	new ReLU(tm, kernelRegistry, "ReLU2"),
	new Dropout(tm, kernelRegistry, "dropout2", 0.3),

	new Linear(tm, kernelRegistry, {
		name: "third",
		inputFeatures: 128,
		outputFeatures: 10,
		useBias: true,
		initializer: heNormal
	}),

	 */
);

const epochs = 10;
const batchSize = 32;


const datasource = new MNISTDatasource();
await datasource
	.load()
	.catch((e: Error) => {
		throw new Error("Failed to load data source: " + e)
	});

datasource.maxTrainSize = 1000;

const optimizer = new SGD(model.parameters(), 0.001, tm, kernelRegistry, batchSize, 1);
optimizer.setSchedule({ type: "cosine", minLr: 0.005, maxSteps: datasource.testImagesCount });

let currentEpoch = 0;
let iterator: MNISTDataSourceIterator = datasource.getTrainIterator(batchSize);

async function train(epoch: number, iterator: MNISTDataSourceIterator) {
	if (iterator.hasNext()) {

		// 1. Zero gradients
		optimizer.zeroGrad();

		// 2. Prepare data
		const data = iterator.next();

		const input = tm.getTensorBuffer(
			"input",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[batchSize, 28 * 28],
			data.data,
		);

		const labelsOneHot = tm.getTensorBuffer(
			"labels_onehot",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[batchSize, 10],
			data.labels
		);

		// 3. Forward (begin scope for transient tensors)
		tm.beginScope("fwd");
		const logits = model.forward(input, true);
		const loss = kernelRegistry.crossEntropy.run(logits, labelsOneHot);

		// 4. Backward (scope set inside topologicalSort)
		topologicalSort(tm, kernelRegistry, loss);

		await GPUEnv.device.queue.onSubmittedWorkDone();

		// 5. Optimize, SGD
		optimizer.step();

		await GPUEnv.device.queue.onSubmittedWorkDone();
	} else {
		iterator = datasource.getTrainIterator(batchSize);
		currentEpoch++;
		onUpdateData(currentEpoch, epochs, iterator.getCurrentIndex(), iterator.getSize());
	}

	if (currentEpoch < epochs) {
		onUpdateData(currentEpoch, epochs, iterator.getCurrentIndex(), iterator.getSize());
		requestAnimationFrame(() => train(currentEpoch, iterator));
	} else {
		requestAnimationFrame(() => test())
	}
}

requestAnimationFrame(() => train(currentEpoch, iterator));

function onUpdateData(epoch: number, epochs: number, current: number, total: number) {
	const node = document.getElementById("out");
	const out  = `Epoch ${epoch}/${epochs} (${current}/${total})`;
	if (node !== null) {
		node.innerHTML = out;
	}
}

let testCorrect = 0;
let testTotal = 0;
let testIterator: MNISTDataSourceIterator | null = null;

async function test() {
	if (testIterator === null) {
		testIterator = datasource.getTestIterator(batchSize);
		testCorrect = 0;
		testTotal = 0;
	}

	if (testIterator.hasNext()) {
		const data = testIterator.next();

		const input = tm.getTensorBuffer(
			"test_input",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[batchSize, 28 * 28],
			data.data,
		);

		// Forward pass (isTraining = false to disable dropout)
		tm.beginScope("test");
		const logits = model.forward(input, false);
		const probs = kernelRegistry.softmax.run(logits);

		// await GPUEnv.device.queue.onSubmittedWorkDone();

		// Read back predictions
		const logitsData = await tm.readBuffer(logits.buffer, logits.sizeInBytes());
		const probsData = await tm.readBuffer(probs.buffer, probs.sizeInBytes());

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
				testCorrect++;
			}
			testTotal++;
		}

		onTestUpdate(testCorrect, testTotal, testIterator.getCurrentIndex(), testIterator.getSize());
		requestAnimationFrame(test);
	} else {

		const accuracy = (testCorrect / testTotal * 100).toFixed(2);
		onTestComplete(testCorrect, testTotal);
		testIterator = null;
	}
}

function onTestUpdate(correct: number, total: number, current: number, size: number) {
	const node = document.getElementById("outtest");
	const accuracy = (correct / total * 100).toFixed(2);
	const out = `Testing: ${current}/${size} - Accuracy: ${accuracy}%`;
	if (node !== null) {
		node.innerHTML = out;
	}
}

function onTestComplete(correct: number, total: number) {
	const node = document.getElementById("outtest");
	const accuracy = (correct / total * 100).toFixed(2);
	const out = `Test complete: ${correct}/${total} (${accuracy}%)`;
	if (node !== null) {
		node.innerHTML = out;
	}
}
