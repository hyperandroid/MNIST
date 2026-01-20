
export function randn() {
	const u1 = 1 - Math.random(); // (0,1]
	const u2 = Math.random();     // [0,1)
	return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 *
 * @param shape
 * @param featuresIn are layer input features count
 * @param w optional weight tensor
 */
export function heNormal(
	shape: number[],
	featuresIn: number,
	w?: Float32Array
): Float32Array {
	const std = Math.sqrt(2 / featuresIn);
	const size = shape.reduce((a, b) => a * b, 1);

	w = w ?? new Float32Array(size);

	for (let i = 0; i < size; i++) {
		w[i] = randn() * std;
	}
	return w;
}

/**
 *
 * @param shape
 * @param fanIn are layer input features count
 * @param w optional weight tensor
 */
export function heUniform(
	shape: number[],
	fanIn: number,
	w?: Float32Array,
): Float32Array {
	const limit = Math.sqrt(6 / fanIn);
	const size = shape.reduce((a, b) => a * b, 1);

	w = w ?? new Float32Array(size);

	for (let i = 0; i < size; i++) {
		w[i] = (Math.random() * 2 - 1) * limit;
	}
	return w;
}
