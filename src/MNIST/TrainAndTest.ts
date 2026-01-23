import {GPUEnv} from "../GPUEnv";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {TensorManager} from "../tensor/TensorManager";
import {MNISTDatasource} from "./MNISTDatasource";
import {MNIST} from "./MNIST";
import {Trainer} from "./Trainer";
import {Tester} from "./Tester";

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
	},
	(imageData: Float32Array, guessed: number, label: number, errorCount: number) => {

		if (errorCount >= 100) {
			return;
		}

		const parent = document.createElement("span");
		parent.style.display = "inline-block";
		parent.style.margin = "2px";

		const px = 5;
		const canvas = document.createElement("canvas");
		canvas.width = 28 * px;
		canvas.height = 28 * px;
		parent.appendChild(canvas);

		const ctx = canvas.getContext("2d");
		if (!ctx) {
			throw new Error("MNISTDatasource: failed to get 2d canvas context");
		}
		for (let r = 0; r < 28; r++) {
			for (let c = 0; c < 28; c++) {
				const index = r * 28 + c;
				const value = Math.floor(imageData[index] * 255);
				ctx.fillStyle = `rgba(${value}, ${value}, ${value}, 1)`;
				ctx.fillRect(c * px, r * px, px, px);
			}
		}

		parent.appendChild(document.createElement("br"));
		parent.appendChild(document.createTextNode(`Guessed: ${guessed}`));
		parent.appendChild(document.createElement("br"));
		parent.appendChild(document.createTextNode(`Expected: ${label}`));

		document.body.appendChild(parent);
	}
);

await tester.initialize();
await train();
