import {LRSchedule, Optimizer} from "./Optimizer";
import {TensorManager} from "../tensor/TensorManager";
import {KernelRegistry} from "../tensor/kernel/KernelRegistry";
import {Tensor} from "../tensor/Tensor";

export class SGD implements Optimizer {
	private baseLr: number;
	private currentLr: number;
	private stepCount = 0;
	private schedule: LRSchedule = { type: "constant" };

	constructor(
		readonly params: Tensor[],
		lr: number,
		private tm: TensorManager,
		private kr: KernelRegistry,
		private batchSize: number,
		private maxGradNorm: number | null = null,
	) {
		this.baseLr = lr;
		this.currentLr = lr;
	}

	step(batchSizeOverride?: number) {
		// Update learning rate based on schedule
		this.updateLearningRate();

		const effectiveBatchSize = batchSizeOverride ?? this.batchSize;

		for (const p of this.params) {
			if (!p.gradient) continue;

			// Clip gradient by norm if maxGradNorm is set
			if (this.maxGradNorm !== null) {
				this.kr.clipGradNorm.run(p.gradient, this.maxGradNorm);
			}

			// p = p - lr * grad
			const update = this.kr.scalarMul.run(p.gradient, -this.currentLr / effectiveBatchSize);
			this.kr.inplaceAdd.run(p, update);
		}

		this.stepCount++;
	}

	private updateLearningRate(): void {
		switch (this.schedule.type) {
			case "constant":
				// No change
				break;

			case "step":
				// Decay by factor every N steps
				if (this.stepCount > 0 && this.stepCount % this.schedule.everyNSteps === 0) {
					this.currentLr *= this.schedule.factor;
				}
				break;

			case "exponential":
				// Exponential decay each step: lr = baseLr * (decayRate ^ step)
				this.currentLr = this.baseLr * Math.pow(this.schedule.decayRate, this.stepCount);
				break;

			case "cosine":
				// Cosine annealing: lr oscillates between baseLr and minLr
				const progress = Math.min(this.stepCount / this.schedule.maxSteps, 1);
				this.currentLr = this.schedule.minLr +
					0.5 * (this.baseLr - this.schedule.minLr) * (1 + Math.cos(Math.PI * progress));
				break;
		}
	}

	zeroGrad() {
		for (const p of this.params) {
			if (p.gradient) {
				this.tm.writeBufferF32(
					p.gradient.buffer,
					new Float32Array(p.gradient.size).fill(0)
				);
			}
		}
	}

	getLearningRate(): number {
		return this.currentLr;
	}

	setLearningRate(lr: number): void {
		this.baseLr = lr;
		this.currentLr = lr;
	}

	setSchedule(schedule: LRSchedule): void {
		this.schedule = schedule;
		// Reset to base learning rate when schedule changes
		this.currentLr = this.baseLr;
	}

	getStepCount(): number {
		return this.stepCount;
	}

	resetStepCount(): void {
		this.stepCount = 0;
		this.currentLr = this.baseLr;
	}
}
