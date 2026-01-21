import {GPUEnv} from "./GPUEnv";
import {KernelRegistry} from "./tensor/kernel/KernelRegistry";
import {TensorManager} from "./tensor/TensorManager";
import {MNISTDatasource, MNISTDataSourceIterator} from "./MNIST/MNISTDatasource";
import {Sequential} from "./layer/Sequential";
import {Linear} from "./layer/Linear";
import {heNormal} from "./math/Utils";
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
);

const optimizer = new SGD(model.parameters(), 0.01, tm, kernelRegistry);

const epochs = 10;
const batchSize = 32;

const datasource = new MNISTDatasource();
await datasource
	.load()
	.catch((e: Error) => {
		throw new Error("Failed to load data source: " + e)
	});


let currentEpoch = 0;
let iterator: MNISTDataSourceIterator = datasource.getTrainIterator(batchSize);

async function oneRun(epoch: number, iterator: MNISTDataSourceIterator) {
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

		// 3. Forward
		const logits = model.forward(input, true);
		const probs = kernelRegistry.softmax.run(logits);
		const loss = kernelRegistry.crossEntropy.run(probs, labelsOneHot);

		// 4. Backward
		topologicalSort(tm, kernelRegistry, loss);

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
		requestAnimationFrame(() => oneRun(currentEpoch, iterator));
	}
}

setTimeout(() => {

	requestAnimationFrame(() => oneRun(currentEpoch, iterator));
}, 5000);

function onUpdateData(epoch: number, epochs: number, current: number, total: number) {
	const node = document.getElementById("out");
	const out  = `Epoch ${epoch}/${epochs} (${current}/${total})`;
	if (node !== null) {
		node.innerHTML = out;
	}
	console.log(out);
}

/*
for (let epoch = 0; epoch < epochs; epoch++) {
	const iterator = datasource.getTrainIterator(batchSize);

	while (iterator.hasNext()) {
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

		// 3. Forward
		const logits = model.forward(input, true);
		const probs = kernelRegistry.softmax.run(logits);
		const loss = kernelRegistry.crossEntropy.run(probs, labelsOneHot);

		// 4. Backward
		topologicalSort(tm, kernelRegistry, loss);

		// 5. Optimize, SGD
		optimizer.step();

		await GPUEnv.device.queue.onSubmittedWorkDone();
	}
}
*/
