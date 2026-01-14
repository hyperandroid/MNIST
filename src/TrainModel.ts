import {Model} from "./Model";
import {GPUEnv} from "./GPUEnv";
import {OpsRegistry} from "./tensorOps/OpsRegistry";
import {TensorManager} from "./TensorManager";

/**
 * Initialize the GPU. If not present, no training will be possible.
 */
await GPUEnv.init()

/**
 * Initialize the tensor operations registry.
 */
const opsRegistry = new OpsRegistry(GPUEnv.device);

const tm = new TensorManager(GPUEnv.device);

/**
 * test code
 */

const M = 4096, K = 4096, N = 128;

const aSize = M * K;
const bSize = K * N;
const cSize = M * N;

// Initialize input tensors
const A = new Float32Array(aSize);
const B = new Float32Array(bSize);

// Simple deterministic values (good for debugging)
for (let i = 0; i < A.length; i++) A[i] = (i % 97) * 0.01;
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
const out = tm.getTensorBuffer(
	"out",
	GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	[M,N],
);

opsRegistry.matmul.run(t0, t1, out);

const output = await tm.readBuffer(out.buffer, out.sizeInBytes());
console.log(output);

/**
 * The train loop will look as follows:
 * + Forward pass
 * + Calculate loss function
 * + Backward pass (gradient descent)
 * + Optimize weights
 */
const model = new Model()

