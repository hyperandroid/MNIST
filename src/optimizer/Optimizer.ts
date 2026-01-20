export interface Optimizer {
	step(): void;
	zeroGrad(): void;
}
