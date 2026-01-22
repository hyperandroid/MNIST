import {GPUEnv} from "./GPUEnv";
import {KernelRegistry} from "./tensor/kernel/KernelRegistry";
import {TensorManager} from "./tensor/TensorManager";
import {MNISTDatasource} from "./MNIST/MNISTDatasource";
import {MNIST} from "./MNIST/MNIST";
import {Trainer} from "./MNIST/Trainer";
import {Tester} from "./MNIST/Tester";

await GPUEnv.init()

const tm = new TensorManager(GPUEnv.device);
const kernelRegistry = new KernelRegistry(GPUEnv.device, tm);

// createa  model,
const mnist = new MNIST(tm, kernelRegistry);
// load pre-trained model. 97.35% accuracy on the test set.
await mnist.readSnapshot();

// create train/test datasource
const datasource = new MNISTDatasource();
await datasource
	.load("data/mnist")
	.catch((e: Error) => {
		throw new Error("Failed to load data source: " + e)
	});

async function train() {

	const trainer = new Trainer(
		tm,
		kernelRegistry,
		mnist,
		datasource,
		() => {
			requestAnimationFrame(() => tester.startTesting());
		},
		(epoch, epochs, current, total) => {
			const node = document.getElementById("out");
			const out = `Epoch ${epoch}/${epochs} (${current}/${total})`;
			if (node !== null) {
				node.innerHTML = out;
			}
		},
	);
	await trainer.initialize();
	trainer.startTraining();
}

const tester = new Tester(
	tm,
	kernelRegistry,
	mnist,
	datasource,
	(correct, total, current, size) => {
		const node = document.getElementById("outtest");
		const accuracy = (correct / total * 100).toFixed(2);
		const out = `Testing: ${current}/${size} - Accuracy: ${accuracy}%`;
		if (node !== null) {
			node.innerHTML = out;
		}
	},
	(correct: number, total: number) => {
		const node = document.getElementById("outtest");
		const accuracy = (correct / total * 100).toFixed(2);
		const out = `Test complete: ${correct}/${total} (${accuracy}%)`;
		if (node !== null) {
			node.innerHTML = out;
		}
	}
);

await tester.initialize();
await train();
