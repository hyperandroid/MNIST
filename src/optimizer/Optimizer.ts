export type LRSchedule =
	| { type: "constant" }
	| { type: "step"; factor: number; everyNSteps: number }
	| { type: "exponential"; decayRate: number }
	| { type: "cosine"; minLr: number; maxSteps: number };

export interface Optimizer {
	step(): void;
	zeroGrad(): void;
	getLearningRate(): number;
	setLearningRate(lr: number): void;
	setSchedule(schedule: LRSchedule): void;
}
