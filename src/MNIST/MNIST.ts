import {Sequential} from "../layer/Sequential";
import {Linear} from "../layer/Linear";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {heUniform} from "../math/Utils";
import {ReLU} from "../layer/ReLU";
import {MNISTDatasource} from "./MNISTDatasource";

export class MNIST {

	readonly model: Sequential;
	readonly firstLayer: Linear;
	readonly secondLayer: Linear;

	constructor(
		readonly tm: TensorManager,
		kernelRegistry: KernelRegistry,
		readonly initializer = heUniform,
	) {
		this.firstLayer = new Linear(tm, kernelRegistry, {
			name: "first",
			inputFeatures: MNISTDatasource.imageSize,
			outputFeatures: 128,
			useBias: true,
			initializer
		});
		this.secondLayer = new Linear(tm, kernelRegistry, {
			name: "second",
			inputFeatures: 128,
			outputFeatures: 10,
			useBias: true,
			initializer
		});

		this.model = new Sequential(
			this.firstLayer,
			new ReLU(tm, kernelRegistry, "ReLU1"),
			this.secondLayer,
		);
	}

	async readSnapshot() {
		const path = "data/trained_768_128_10";

		const firstLayerWeightsR = await fetch(`${path}/model-first_weights.bin`);
		const firstWeights = new Float32Array(await firstLayerWeightsR.arrayBuffer());
		const firstLayerBiasR = await fetch(`${path}/model-first_bias.bin`);
		const firstBias = new Float32Array(await firstLayerBiasR.arrayBuffer());
		this.tm.writeBufferF32(this.firstLayer.parameters()[0].buffer, firstWeights);
		this.tm.writeBufferF32(this.firstLayer.parameters()[1].buffer, firstBias);

		const secondLayerWeightsR = await fetch(`${path}/model-second_weights.bin`);
		const secondWeights = new Float32Array(await secondLayerWeightsR.arrayBuffer());
		const secondLayerBiasR = await fetch(`${path}/model-second_bias.bin`);
		const secondBias = new Float32Array(await secondLayerBiasR.arrayBuffer());
		this.tm.writeBufferF32(this.secondLayer.parameters()[0].buffer, secondWeights);
		this.tm.writeBufferF32(this.secondLayer.parameters()[1].buffer, secondBias);
	}

	async restart() {
		this.model.zeroGrad(this.tm);

		// Clear bias
		this.tm.zeros(this.firstLayer.parameters()[1]);
		this.tm.zeros(this.secondLayer.parameters()[1]);

		// Reinitialize weights
		const s0 = this.firstLayer.parameters()[0].shape;
		this.tm.writeBufferF32(this.firstLayer.parameters()[0].buffer, heUniform(s0, s0[0]));
		const s1 = this.secondLayer.parameters()[0].shape;
		this.tm.writeBufferF32(this.secondLayer.parameters()[0].buffer, heUniform(s1, s1[0]));
	}
}
