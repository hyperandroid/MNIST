import {GPUEnv} from "../../GPUEnv";
import {KernelRegistry} from "../../tensor/kernel/KernelRegistry";
import {TensorManager} from "../../tensor/TensorManager";
import {MNISTDatasource} from "../MNISTDatasource";
import {MNIST} from "../MNIST";
import {PaintLayer} from "./PaintLayer";
import {LayerInputs} from "./LayerInputs";
import {Tensor} from "../../tensor/Tensor";

await GPUEnv.init();

const tm = new TensorManager(GPUEnv.device);
const kernelRegistry = new KernelRegistry(GPUEnv.device, tm);

// create model,
const mnist = new MNIST(tm, kernelRegistry);
// load pre-trained model. 97.35% accuracy on the test set.
const parameters = await mnist.readSnapshot();
generateParameterImages(parameters);

// create train/test datasource
const datasource = new MNISTDatasource();
await datasource
	.load("data/mnist")
	.catch((e: Error) => {
		throw new Error("Failed to load data source: " + e)
	});

const emptyData = new Float32Array(10);
let working = false;

const painter = new PaintLayer(
	async (data: Float32Array) => {
		await GPUEnv.device.queue.onSubmittedWorkDone();
			if (working) {
				return;
			}
			working = true;
			await test(data)
			working = false;
	},
	async () => {
		setProbability(emptyData);
		layersActivations.render([], []);
	});

const layersActivations = new LayerInputs();

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

	const probsData = await tm.readBuffer(probs.buffer, probs.sizeInBytes());
	setProbability(probsData);

	// read back layers inputs.
	const outputs: Float32Array[] = [];
	for(const layer of mnist.model.layers) {
		const it = layer.inputTensor!;
		const buffer = await tm.readBuffer(it.buffer, it.sizeInBytes());
		outputs.push(buffer);
	}
	outputs.push(probsData);

	const names = mnist.model.layers.map(l => `${l.name} [${l.inputTensor?.shape}]`);
	names.push("Predictions. Shape [1,10]. Digit 0..9")
	layersActivations.render(outputs, names);
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

function generateParameterImages(data: Float32Array[]) {
	generateParameterImage(data[0], 768, 128, data[1], 128, 1, "First Layer. Dense [768->128]");
	generateParameterImage(data[2], 128, 10, data[3], 10, 1, "Output Layer. Dense [128->10]" );
}

function generateParameterImage(
	weights: Float32Array, rowsw: number, colsw: number,
	bias: Float32Array, rowsb: number, colsb: number,
	title: string,
) {
	const anchor = document.getElementById("parameters-container")!;
	const div = document.createElement("div");
	div.className = "parameter-images"
	anchor.appendChild(div);

	const h4 = document.createElement("h4");
	h4.innerHTML = title;
	div.appendChild(h4);

	generateImage(div, weights, rowsw, colsw, "Weights");
	generateImage(div, bias, rowsb, colsb, "Bias");
}

function generateImage(div: HTMLDivElement, parameter: Float32Array, cols: number, rows: number, title: string) {
	const canvas = document.createElement("canvas");

	const scale = cols < 768 ? 10 : 2;
	canvas.width = cols * scale;
	canvas.height = rows * scale;
	const ctx = canvas.getContext("2d")!;

	const container = document.createElement("div");
	container.className = "parameter-image";
	container.appendChild(document.createTextNode(`${title}`));
	container.appendChild(canvas);

	div.appendChild(container);


	const p = normalize(parameter);

	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			const index = r * cols + c;
			const col = Math.floor(p[index] * 255);
			ctx.fillStyle = `rgba(${col}, ${col}, ${col*.8}, 1)`;
			ctx.fillRect(c * scale, r * scale, scale, scale);
		}
	}
}

function normalize(i: Float32Array): Float32Array {
	const data = new Float32Array(i.length);
	data.set(i,0);

	let max = -Infinity;
	let min = Infinity;
	for(let i =0; i<data.length; i++) {
		if (data[i] > max) max = data[i];
		if (data[i] < min) min = data[i];
	}
	for(let i =0; i<data.length; i++) {
		data[i] = ((data[i] - min) / (max - min));
	}

	return data;
}
