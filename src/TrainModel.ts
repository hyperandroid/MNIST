import {Model} from "./Model";
import {GPUEnv} from "./GPUEnv";
import {KernelRegistry} from "./tensorOps/KernelRegistry";
import {TensorManager} from "./TensorManager";

/**
 * Initialize the GPU. If not present, no training will be possible.
 */
await GPUEnv.init()

const tm = new TensorManager(GPUEnv.device);
const opsRegistry = new KernelRegistry(GPUEnv.device, tm);


/**
 * test code
 */

const M = 4096, K = 4096, N = 128;

const aSize = M * K;
const bSize = K * N;

// Initialize input tensors
const A = new Float32Array(aSize);
const B = new Float32Array(bSize);

// Simple deterministic values (good for debugging)
for (let i = 0; i < A.length; i++) A[i] = (i % 97) * 0.01 * (Math.random()<.5 ? 1 : -1);
for (let i = 0; i < B.length; i++) B[i] = (i % 89) * 0.02;

const t0 = tm.getTensorBuffer(
	"t0",
	GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	[M,K],
	A,
);
const t1 = tm.getTensorBuffer(
	"t1",
	GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	[K,N],
	B,
);

const out = opsRegistry.relu.run(
	opsRegistry.matmul.run(t0, t1)
);

const output2 = await tm.readBuffer(out.buffer, out.sizeInBytes());
console.log(output2);

/**
 * The train loop will look as follows:
 * + Forward pass
 * + Calculate loss function
 * + Backward pass (gradient descent)
 * + Optimize weights
 */
const model = new Model()

