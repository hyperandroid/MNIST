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

const timerElement = document.getElementById("timer")!;
let timerInterval: number | null = null;
let startTime: number = 0;

function formatTime(ms: number): string {
	const minutes = Math.floor(ms / 60000);
	const seconds = Math.floor((ms % 60000) / 1000);
	const millis = ms % 1000;
	return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}.${millis.toString().padStart(3, "0")}`;
}

function startTimer() {
	startTime = performance.now();
	timerInterval = window.setInterval(() => {
		const elapsed = performance.now() - startTime;
		timerElement.textContent = formatTime(Math.floor(elapsed));
	}, 10);
}

function stopTimer() {
	if (timerInterval !== null) {
		clearInterval(timerInterval);
		timerInterval = null;
		const elapsed = performance.now() - startTime;
		timerElement.textContent = formatTime(Math.floor(elapsed));
	}
}

async function train(epochs: number, batchSize: number) {

	const trainer = new Trainer(
		tm,
		kernelRegistry,
		mnist,
		datasource,
		() => {
			stopTimer();
			startBtn.textContent = "Done";
			requestAnimationFrame(() => tester.startTesting());
		},
		(epoch, epochs, current, total) => {
			const node = document.getElementById("out");
			const out = `Epoch ${epoch}/${epochs} (${current}/${total})`;
			if (node !== null) {
				node.innerHTML = out;
			}
		},
		epochs,
		batchSize,
	);
	await trainer.initialize();
	startTimer();
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

		const container = document.getElementById("errors-container");
		if (!container) return;

		const parent = document.createElement("div");
		parent.className = "error-sample";

		const px = 4;
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

		const label1 = document.createElement("span");
		label1.textContent = `Predicted: ${guessed}`;
		parent.appendChild(label1);

		const label2 = document.createElement("span");
		label2.textContent = `Actual: ${label}`;
		parent.appendChild(label2);

		container.appendChild(parent);
	}
);

await tester.initialize();

const startBtn = document.getElementById("start-btn") as HTMLButtonElement;
const epochsSelect = document.getElementById("epochs-select") as HTMLSelectElement;
const batchSizeSelect = document.getElementById("batchsize-select") as HTMLSelectElement;

startBtn.addEventListener("click", async () => {
	const epochs = parseInt(epochsSelect.value, 10);
	const batchSize = parseInt(batchSizeSelect.value, 10);

	startBtn.disabled = true;
	epochsSelect.disabled = true;
	batchSizeSelect.disabled = true;
	startBtn.textContent = "Training...";

	await train(epochs, batchSize);
});
