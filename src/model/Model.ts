
export interface Model {
	/**
	 * The train loop will look as follows:
	 * + Forward pass
	 * + Calculate loss function
	 * + Backward pass (gradient descent)
	 * + Optimize weights
	 */
	train(epochs: number, batchSize: number): void;

	test(batchSize: number): void;

	forward(batchSize: number): void;

	backward(batchSize: number): void;
}
