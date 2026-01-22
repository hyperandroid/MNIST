import {GPUEnv} from "./GPUEnv";
import {KernelRegistry} from "./tensor/kernel/KernelRegistry";
import {TensorManager} from "./tensor/TensorManager";
import {MNISTDatasource, MNISTDataSourceIterator} from "./MNIST/MNISTDatasource";
import {SGD} from "./optimizer/SGD";
import {computeBackwardPass} from "./autograd/BackwardPass";
import {MNIST} from "./MNIST/MNIST";

await GPUEnv.init()

const epochs = 15;
const batchSize = 32;
const tm = new TensorManager(GPUEnv.device);
const kernelRegistry = new KernelRegistry(GPUEnv.device, tm);
const mnist = new MNIST(tm, kernelRegistry);
await mnist.readSnapshot();
const model = mnist.model;

const datasource = new MNISTDatasource();
await datasource
	.load("data/mnist")
	.catch((e: Error) => {
		throw new Error("Failed to load data source: " + e)
	});

const optimizer = new SGD(model.parameters(), 0.05, tm, kernelRegistry, batchSize);
const trainSize = Math.min(datasource.trainImagesCount, datasource.maxTrainSize);
const stepsPerEpoch = Math.ceil(trainSize / batchSize);
optimizer.setSchedule({ type: "cosine", minLr: 0.001, maxSteps: stepsPerEpoch * epochs });

let currentEpoch = 0;
let iterator: MNISTDataSourceIterator = datasource.getTrainIterator(batchSize);

async function snapshot() {
	for(const parameter of model.parameters()) {
		const buffer = await tm.readBuffer(parameter.buffer, parameter.sizeInBytes());
		const blob = new Blob([buffer], {type: "application/octet-stream"});
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = `model-${currentEpoch}-${parameter.name}.bin`;
		a.click();
		URL.revokeObjectURL(url);
	}
}

async function sync() {
	await GPUEnv.device.queue.onSubmittedWorkDone();
}

async function train(epoch: number, iterator: MNISTDataSourceIterator) {
	if (iterator.hasNext()) {

		// 1. Zero gradients
		optimizer.zeroGrad();
		await GPUEnv.device.queue.onSubmittedWorkDone();

		// 2. Prepare data
		const data = iterator.next();
		const currentBatchSize = data.size;

		const input = tm.getTensorBuffer(
			"input",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[currentBatchSize, 28 * 28],
			data.data,
		);

		const labelsOneHot = tm.getTensorBuffer(
			"labels_onehot",
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			[currentBatchSize, 10],
			data.labels
		);

		// 3. Forward (begin scope for transient tensors)
		tm.beginScope("fwd");
		const logits = model.forward(input, true);
		const loss = kernelRegistry.crossEntropy.run(logits, labelsOneHot);

		// 4. Backward (scope set inside topologicalSort)
		computeBackwardPass(tm, kernelRegistry, loss);

		sync();

		// 5. Optimize, SGD
		optimizer.step(currentBatchSize);

		sync();

	} else {
		iterator = datasource.getTrainIterator(batchSize);
		currentEpoch++;
		onUpdateData(currentEpoch, epochs, iterator.getCurrentIndex(), iterator.getSize());
		await snapshot();
	}

	if (currentEpoch < epochs) {
		onUpdateData(currentEpoch, epochs, iterator.getCurrentIndex(), iterator.getSize());
		requestAnimationFrame(() => train(currentEpoch, iterator));
	} else {
		await snapshot();
		requestAnimationFrame(() => test())
	}
}

//requestAnimationFrame(() => train(currentEpoch, iterator));

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

		// Read back predictions
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
		onTestComplete(testCorrect, testTotal);
		testIterator = null;
	}
}

requestAnimationFrame(() => test());

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
