import {GPUEnv} from "../../GPUEnv";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";
import {TensorManager} from "../../tensor/TensorManager";
import {MNISTDatasource} from "../MNISTDatasource";
import {MNIST} from "../MNIST";
import {PaintLayer} from "./PaintLayer";

await GPUEnv.init();

const tm = new TensorManager(GPUEnv.device);
const kernelRegistry = new KernelRegistry(GPUEnv.device, tm);

// create model,
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

const emptyData = new Float32Array(10);

const painter = new PaintLayer(
	async (data: Float32Array) => {
		await test(data);
	},
	async () => {
		setProbability(emptyData);
	});

(function() {

	const wrapper0 = document.getElementById("row")!;

	const wrapper = document.createElement("span");
	wrapper.style.display = "inline-flex";
	wrapper.style.flexDirection = "column";
	wrapper.style.gap = "6px";

	let index = 0;
	Array.from({ length: 10 }, () => {

		const line = document.createElement("span");
		line.classList.add("probability");
		line.id = `probability-${index}`;

		const slider = document.createElement("input");
		slider.style.pointerEvents = "none";
		slider.type = "range";
		slider.min = "0";
		slider.max = "100";
		slider.value = "0";
		slider.id = `probability-slider-${index}`;

		line.appendChild(slider);
		line.appendChild(document.createTextNode(`${index}`));

		const percentage = document.createElement("span");
		percentage.id = `probability-percentage-${index}`;
		line.appendChild(percentage);

		wrapper.appendChild(line);

		index ++;

		return slider;
	});

	wrapper0.appendChild(wrapper);
	document.body.appendChild(wrapper0);
})();

async function test(data: Float32Array) {
	const input = tm.getTensorBuffer(
		"test_input",
		GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		[1, data.length],
		data,
	);

	tm.beginScope("test");
	const logits = mnist.model.forward(input, false);
	const probs = kernelRegistry.softmax.run(logits);

	await GPUEnv.device.queue.onSubmittedWorkDone();

	const probsData = await tm.readBuffer(probs.buffer, probs.sizeInBytes());
	setProbability(probsData);
}

function setProbability(probsData: Float32Array) {
	let maxProb = -Infinity;
	let predicted = 0;
	for (let j = 0; j < 10; j++) {
		const prob = probsData[j];
		if (prob > maxProb) {
			maxProb = prob;
			predicted = j;
		}

		const node = document.getElementById(`probability-${j}`);
		if (node !== null) {
			node.style.backgroundColor = "default";
		}

		const nodes = document.getElementById(`probability-slider-${j}`);
		if (nodes !== null) {
			(nodes as any).value = (100*prob).toFixed(2);
		}

		const percentage = (prob * 100).toFixed(2);
		const node2 = document.getElementById(`probability-percentage-${j}`);
		if (node2 !== null) {
			node2.innerHTML = `${percentage}%`;
		}
	}

	for(let j = 0; j<10; j++) {
		const node = document.getElementById(`probability-percentage-${j}`)!;
		node.style.backgroundColor = (j===predicted && probsData[predicted]>0)
			? "green"
			: "rgba(0,0,0,0)";
	}
}
