import {Sequential} from "../layer/Sequential";
import {Linear} from "../layer/Linear";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {heUniform} from "../math/Utils";
import {ReLU} from "../layer/ReLU";

export class MNIST {

	readonly model: Sequential;

	constructor(
		readonly tm: TensorManager,
		kernelRegistry: KernelRegistry,
	) {
		this.model = new Sequential(
			new Linear(tm, kernelRegistry, {
				name: "first",
				inputFeatures: 28 * 28,
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
		);
	}

	async readSnapshot() {

		const path = "data/trained";

		const firstLayerWeightsR = await fetch(`${path}/model-first_weights.bin`);
		const firstWeights = new Float32Array(await firstLayerWeightsR.arrayBuffer());
		const firstLayerBiasR = await fetch(`${path}/model-first_bias.bin`);
		const firstBias = new Float32Array(await firstLayerBiasR.arrayBuffer());
		this.tm.writeBufferF32(this.model.layers[0].parameters()[0].buffer, firstWeights);
		this.tm.writeBufferF32(this.model.layers[0].parameters()[1].buffer, firstBias);

		const secondLayerWeightsR = await fetch(`${path}/model-second_weights.bin`);
		const secondWeights = new Float32Array(await secondLayerWeightsR.arrayBuffer());
		const secondLayerBiasR = await fetch(`${path}/model-second_bias.bin`);
		const secondBias = new Float32Array(await secondLayerBiasR.arrayBuffer());
		this.tm.writeBufferF32(this.model.layers[2].parameters()[0].buffer, secondWeights);
		this.tm.writeBufferF32(this.model.layers[2].parameters()[1].buffer, secondBias);
	}
}
