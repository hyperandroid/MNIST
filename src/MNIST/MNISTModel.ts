import {TensorManager} from "../tensor/TensorManager";
import {MNISTDatasource} from "./MNISTDatasource";
import {Model} from "../model/Model";


/**
 * MNIST model.
 *
 * This model will use:
 * - RELU for activation function
 * - Cross Entropy loss as Loss function
 * - SoftMax as Output function
 *
 * The model will be trained on GPU using WebGPU.
 *
 * The default batch size is 32.
 * The model Will run a default of 32 epochs.
 */
export class MNISTModel implements Model {

	constructor(
		readonly device: GPUDevice,
		readonly tm: TensorManager,
		readonly batchSize: number = 32,
		readonly epochs: number = 32,
		readonly learningRate: number = 0.001,
		readonly datasource: MNISTDatasource,
	) {

	}

	forward(batchSize: number): void {

    }
    backward(batchSize: number): void {

    }

	train(epochs: number = 32, batchSize: number = 32) {

	}

	test(batchSize: number = 32) {
	}
}
